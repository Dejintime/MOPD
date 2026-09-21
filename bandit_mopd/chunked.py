"""Bounded-memory, full-normalization OPD for resident CPU teachers.

Cache hidden states, never T x vocab distributions. Routing statistics and
selected targets are evaluated in token blocks. The student output head uses
an explicit chain-rule backward so block graphs are freed before the backbone
backward, including when embeddings and output weights are tied.
"""
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .core import mixture_log_probs
from .context import paper_context, measured_costs, routing_costs
from .batching import batch_settings


@torch.no_grad()
def response_hidden(model, ids, prompt_length, chunk_size=512, progress=None):
    """Causal block prefill with a full-context KV cache; no context truncation."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    cache, chunks = None, []
    # Position prompt_length-1 predicts the first response token.
    stop = ids.shape[1]-1
    for start in range(0, stop, chunk_size):
        end = min(start+chunk_size, stop)
        out = model.model(input_ids=ids[:, start:end],
            attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
            past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        offset = max(prompt_length-1-start, 0)
        if offset < end-start:
            chunks.append(out.last_hidden_state[0, offset:].detach().cpu().clone())
        if progress and (end % 4096 == 0 or end == stop):
            progress(end, stop)
    return torch.cat(chunks)


@torch.no_grad()
def projected_log_probs(model, hidden):
    head = model.get_output_embeddings()
    return head(hidden.to(head.weight.device)).float().log_softmax(-1).cpu()


def compact_target(teachers, student_lp, weights, epsilon, top_k):
    """Exact mixture values on the deduplicated teacher top-k union."""
    if top_k < 1:
        raise ValueError('Chunked backend requires a positive top_k')
    k = min(top_k, teachers.shape[-1])
    ids = teachers.topk(k, dim=-1).indices.permute(1, 0, 2).flatten(1).sort(-1).values
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, 1:] = ids[:, 1:] != ids[:, :-1]
    raw_q = mixture_log_probs(teachers, weights)
    target = torch.logaddexp(raw_q + np.log1p(-epsilon), student_lp + np.log(epsilon)) if epsilon else raw_q
    return ids, valid, target.gather(1, ids), raw_q


def compact_loss(log_probs, target, ids, valid):
    p = log_probs.gather(1, ids.to(log_probs.device))
    q = target.to(p.device)
    terms = p.exp()*(p-q)-p.exp()+q.exp()
    return (terms*valid.to(p.device)).sum()


def student_loss(student, sample, chunk_size=128, backward=False, scale=1.0):
    """Exact full-parameter gradient of the existing normalized top-k loss."""
    device = next(student.parameters()).device
    ids = sample['ids'].to(device)
    count = len(sample['response_ids'])
    if backward:
        full = student.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        hidden = full[0, sample['prompt_length']-1:-1]
        leaf = hidden.detach().requires_grad_(True)
    else:
        hidden = response_hidden(student, ids, sample['prompt_length'])
        leaf = hidden
    total = 0.0
    with torch.set_grad_enabled(backward):
        for start in range(0, count, chunk_size):
            end = min(start+chunk_size, count)
            head = student.get_output_embeddings()
            lp = head(leaf[start:end].to(device)).float().log_softmax(-1)
            loss = compact_loss(lp, sample['target'][start:end], sample['union_ids'][start:end],
                                sample['union_valid'][start:end])/count
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite chunked student loss')
            total += loss.detach().item()
            if backward:
                (loss*scale).backward()
            del lp, loss
        if backward:
            hidden.backward(leaf.grad)
    return total


def student_loss_batch(student, samples, chunk_size=128, backward=False, scale=1.0, pad_token_id=0):
    """A true padded backbone microbatch with exact per-sequence mean losses.

    Output projections are still chunked. scale=1/effective_batch gives the
    same gradient whether examples are batched or accumulated sequentially.
    """
    if not samples: raise ValueError('Empty student microbatch')
    if not backward:
        return [student_loss(student,s,chunk_size) for s in samples]
    if len(samples) == 1:
        return [student_loss(student,samples[0],chunk_size,True,scale)]
    device = next(student.parameters()).device
    lengths = [s['ids'].shape[1] for s in samples]
    width = max(lengths)
    ids = torch.full((len(samples),width),pad_token_id,dtype=torch.long,device=device)
    mask = torch.zeros_like(ids)
    for i,(sample,length) in enumerate(zip(samples,lengths)):
        ids[i,:length]=sample['ids'][0].to(device)
        mask[i,:length]=1
    positions=(mask.cumsum(-1)-1).clamp_min(0)
    full=student.model(input_ids=ids,attention_mask=mask,position_ids=positions,
                       use_cache=False).last_hidden_state
    leaf=full.detach().requires_grad_(True)
    totals=[]
    for i,sample in enumerate(samples):
        count=len(sample['response_ids'])
        offset=sample['prompt_length']-1
        total=0.0
        for start in range(0,count,chunk_size):
            end=min(start+chunk_size,count)
            lp=student.get_output_embeddings()(leaf[i,offset+start:offset+end]).float().log_softmax(-1)
            loss=compact_loss(lp,sample['target'][start:end],sample['union_ids'][start:end],
                               sample['union_valid'][start:end])/count
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite resident batch loss')
            total+=loss.detach().item()
            (loss*scale).backward()
            del lp,loss
        totals.append(total)
    full.backward(leaf.grad)
    return totals


def backward_resident_batch(student, samples, cfg, pad_token_id=0):
    if len(samples) != batch_settings(cfg): raise ValueError('Incomplete effective batch')
    backward_size = cfg.get('student_backward_batch_size', cfg['micro_batch_size'])
    if type(backward_size) is not int or not 1 <= backward_size <= cfg['micro_batch_size']:
        raise ValueError('student_backward_batch_size must be between 1 and micro_batch_size')
    result=[]
    for start in range(0,len(samples),cfg['micro_batch_size']):
        stop = start + cfg['micro_batch_size']
        for substart in range(start, stop, backward_size):
            result.extend(student_loss_batch(student,samples[substart:min(substart+backward_size,stop)],
                cfg.get('logit_chunk_tokens',128),backward=True,scale=1/len(samples),pad_token_id=pad_token_id))
    return result


class ResidentTeachers:
    def __init__(self, cfg, models=None):
        if cfg['teacher_device'] != 'cpu':
            raise ValueError('Resident chunked backend requires CPU teachers')
        batch_settings(cfg)
        if cfg['top_k'] < 1 or not 0 <= cfg['epsilon'] < 1:
            raise ValueError('Invalid chunked target settings')
        self.cfg = cfg
        self.prefill_chunk = cfg.get('prefill_chunk_tokens', 512)
        self.logit_chunk = cfg.get('logit_chunk_tokens', 128)
        if min(self.prefill_chunk, self.logit_chunk) < 1:
            raise ValueError('Token chunk sizes must be positive')
        self.models = [] if models is None else list(models)
        if models is None:
            for spec in cfg['teachers']:
                print(f"Loading resident CPU teacher {spec['name']} ({cfg['teacher_dtype']})", flush=True)
                model = AutoModelForCausalLM.from_pretrained(spec['path'], local_files_only=True,
                    dtype=getattr(torch, cfg['teacher_dtype']), device_map='cpu', attn_implementation='sdpa')
                self.models.append(model.requires_grad_(False).eval())
        if len(self.models) != len(cfg['teachers']):
            raise ValueError('Teacher model/spec count mismatch')
        for model in self.models:
            model.requires_grad_(False).eval()

    def prepare(self, student, tokenizer, rows, cfg, rollout, domain_model):
        if len(rows) != batch_settings(cfg): raise ValueError('Incomplete resident teacher batch')
        samples=[self._prepare_one(student,tokenizer,row,cfg,rollout,domain_model) for row in rows]
        contexts=np.mean([s['contexts'] for s in samples],axis=0)
        kls=np.asarray([s['kls'] for s in samples])
        similarity=np.mean([s['similarity'] for s in samples],axis=0)
        selection_costs=np.mean([s['routing_costs'] for s in samples],axis=0)
        costs=[{**samples[0]['costs'][i], **{key:sum(s['costs'][i][key] for s in samples)
                for key in ('prefill_seconds','load_and_prefill_seconds')}} for i in range(len(self.models))]
        return samples,contexts,kls,similarity,selection_costs,costs

    def _prepare_one(self, student, tokenizer, row, cfg, rollout, domain_model):
        tick = time.perf_counter()
        print(json.dumps({'event':'rollout_started','prompt_id':row.get('id'),'domain':row['domain']}),flush=True)
        ids, prompt, text, truncated = rollout(student, tokenizer, row, cfg)
        rollout_seconds = time.perf_counter()-tick
        print(json.dumps({'event':'rollout_completed','prompt_id':row.get('id'),
            'seconds':rollout_seconds,'response_tokens':ids.shape[1]-prompt,'truncated':truncated}),flush=True)
        sample = {'row':row, 'ids':ids.cpu(), 'prompt_length':prompt,
                  'response_ids':ids[0, prompt:].cpu(), 'response':text, 'truncated':truncated,
                  'rollout_seconds':rollout_seconds}
        if cfg.get('save_pending_rollouts'):
            path = Path(cfg['output'])/'pending-rollout.json'
            path.write_text(json.dumps({'prompt_id':row['id'],'ids':ids.cpu().tolist(),
                'prompt_length':prompt,'response':text,'truncated':truncated},ensure_ascii=False))
        print(f"Chunked candidate scoring: {len(sample['response_ids'])} response tokens", flush=True)
        sample['student_hidden'] = response_hidden(student, ids, prompt, self.prefill_chunk)
        sample['teacher_hidden'], costs = [], []
        for spec, teacher in zip(cfg['teachers'], self.models):
            tick = time.perf_counter()
            print(f"CPU prefill {spec['name']}", flush=True)
            h = response_hidden(teacher, ids, prompt, self.prefill_chunk,
                lambda n,total: print(f"CPU prefill {spec['name']}: {n}/{total}", flush=True))
            sample['teacher_hidden'].append(h)
            elapsed = time.perf_counter()-tick
            costs.append({'prefill_seconds':elapsed, 'load_and_prefill_seconds':elapsed,
                          'device':'cpu', 'dtype':str(next(teacher.parameters()).dtype),
                          'resident':True})
        n, t = len(self.models), len(sample['response_ids'])
        kls, entropies, observed = np.zeros(n), np.zeros(n), np.zeros(n)
        similarities = np.zeros((n,n))
        observed_student = 0.0
        for start in range(0, t, self.logit_chunk):
            end = min(start+self.logit_chunk,t)
            p = projected_log_probs(student, sample['student_hidden'][start:end])
            teachers = []
            for i, model in enumerate(self.models):
                tick = time.perf_counter()
                teachers.append(projected_log_probs(model, sample['teacher_hidden'][i][start:end]))
                costs[i]['prefill_seconds'] += time.perf_counter()-tick
                costs[i]['load_and_prefill_seconds'] = costs[i]['prefill_seconds']
            response = sample['response_ids'][start:end, None]
            observed_student += p.gather(1,response).sum().item()/t
            for i, lp in enumerate(teachers):
                kls[i] += (p.exp()*(p-lp)).sum().item()/t
                entropies[i] -= (lp.exp()*lp).sum().item()/t
                observed[i] += lp.gather(1,response).sum().item()/t
            for i,j in itertools.combinations(range(n),2):
                similarities[i,j] += ((teachers[i]+teachers[j])*.5).exp().sum().item()/t
            if end % 4096 == 0 or end == t:
                print(f"Full-vocabulary routing statistics: {end}/{t}", flush=True)
        similarities += similarities.T
        np.fill_diagonal(similarities,1.)
        selection_costs = routing_costs(measured_costs(costs), cfg)
        contexts = paper_context(domain_model.predict(row['messages'], cfg['teachers']),
            observed, kls, entropies, observed_student, similarities, selection_costs)
        sample.update(contexts=contexts,kls=kls.tolist(),costs=costs,similarity=similarities,
                      routing_costs=selection_costs)
        return sample

    def targets(self, student, samples, chosen, weights, cfg):
        for sample in samples:
            union_ids, valid, targets = [], [], []
            r_distill = r_stability = 0.0
            t = len(sample['response_ids'])
            for start in range(0,t,self.logit_chunk):
                end = min(start+self.logit_chunk,t)
                p = projected_log_probs(student,sample['student_hidden'][start:end])
                teachers = torch.stack([projected_log_probs(self.models[i],
                    sample['teacher_hidden'][i][start:end]) for i in chosen])
                ix, mask, q, raw = compact_target(teachers,p,weights,cfg['epsilon'],cfg['top_k'])
                union_ids.append(ix); valid.append(mask); targets.append(q)
                r_distill += (raw-p).gather(1,sample['response_ids'][start:end,None]).sum().item()/t
                r_stability -= (p.exp()*(p-raw)).sum().item()/t
            sample.update(union_ids=torch.cat(union_ids),union_valid=torch.cat(valid),
                          target=torch.cat(targets),r_distill=r_distill,r_stability=r_stability)
            del sample['student_hidden'], sample['teacher_hidden']
