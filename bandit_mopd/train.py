"""Full-parameter Bandit-MOPD with sequential CPU or CUDA teachers.

Full candidate prefills provide current-context features and an exact KL gate.
They are explicitly timed; only sparse serving can demonstrate cost savings.
Pilot configs combine verifier-based probe task gain with dense proxy rewards.
Legacy smoke configs retain dense-only rewards (lambda_task=0).
"""
import argparse
from datetime import datetime, timezone
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .core import CombinatorialLinUCB, mixture_log_probs, reverse_kl_loss, distribution_similarity, sequence_weights
from .context import CONTEXT_FEATURES, PromptDomainModel, configure_context, paper_context, measured_costs, routing_costs
from .data import identity, read_jsonl, read_training_prompts
from .optimizer import CPUAdamW
from .preflight import inspect, sha256
from .devices import synchronize, release_cache, peak_gpu_memory
from .probes import ProbeEvaluator
from .prompting import render_rollout_prompt
from .batching import batch_settings, response_log_probs_batch
from .checkpoints import configure_checkpointing, disk_reserve_bytes, storage_parent, should_save, save_checkpoint, link_final


def log_probs(model, ids, prompt_length):
    # Logit at prefix_length-1 predicts the FIRST generated response token.
    count = ids.shape[1] - prompt_length
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, logits_to_keep=count+1)
    return out.logits[0, :-1].float().log_softmax(-1)


def score_teacher(spec, ids, prompt_length, cfg):
    """Single-trajectory compatibility wrapper."""
    scores, costs = score_teacher_batch(spec, [{'ids': ids, 'prompt_length': prompt_length}], cfg)
    return scores[0], costs[0]


def score_teacher_batch(spec, samples, cfg):
    """Load one teacher once, then score all trajectories sequentially."""
    device = cfg['teacher_device']
    dtype = getattr(torch, cfg.get('teacher_dtype',
                                  'float32' if device == 'cpu' else 'bfloat16'))
    tick = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(spec['path'], local_files_only=True,
        dtype=dtype, device_map=device, attn_implementation='sdpa')
    try:
        teacher.requires_grad_(False).eval()
        synchronize(device)
        loaded = time.perf_counter()
        load_seconds = loaded-tick
        scores, costs = [], []
        for sample in samples:
            start = time.perf_counter()
            with torch.no_grad():
                lp = log_probs(teacher, sample['ids'].to(device), sample['prompt_length']).cpu()
            synchronize(device)
            compute_seconds = time.perf_counter()-start
            scores.append(lp)
            costs.append({'prefill_seconds': compute_seconds,
                          'load_and_prefill_seconds': compute_seconds+load_seconds/len(samples),
                          'device': device, 'dtype': str(dtype)})
        return scores, costs
    finally:
        del teacher
        gc.collect()
        release_cache(device)


def rollout(model, tokenizer, row, cfg, greedy=False):
    prompt = render_rollout_prompt(tokenizer, row['messages'], cfg, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
    if ids.shape[1] > cfg['max_prompt_tokens']:
        raise ValueError('Prompt over budget; filter explicitly instead of truncating it')
    ids = ids.to(cfg['student_device'])
    kwargs = {'max_new_tokens': cfg['max_new_tokens'], 'do_sample': not greedy,
              'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.eos_token_id,
              'use_cache': True, 'top_k': 0, 'top_p': 1.0}
    if not greedy:
        kwargs['temperature'] = cfg['temperature']
    with torch.no_grad():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
    response = generated[0, ids.shape[1]:]
    if not len(response):
        raise RuntimeError('Empty rollout')
    return generated, ids.shape[1], tokenizer.decode(response, skip_special_tokens=True), bool(response[-1] != tokenizer.eos_token_id)


def prepare_batch(student, tokenizer, rows, cfg, domain_model, rollout_fn=None):
    """Collect a fixed pre-update batch and teacher statistics on CPU."""
    rollout_fn = rollout_fn or rollout
    samples = []
    for row in rows:
        ids, prompt_length, text, truncated = rollout_fn(student, tokenizer, row, cfg)
        with torch.no_grad():
            student_lp = log_probs(student, ids, prompt_length).detach().cpu()
        response_ids = ids[0, prompt_length:].cpu()
        samples.append({'row': row, 'ids': ids.cpu(), 'prompt_length': prompt_length,
            'response_ids': response_ids, 'response': text, 'truncated': truncated,
            'student_lp': student_lp, 'teachers': [], 'contexts': [], 'kls': [], 'costs': []})
    for spec in cfg['teachers']:
        print(f"scoring {spec['name']} on {len(samples)} trajectories", flush=True)
        scores, costs = score_teacher_batch(spec, samples, cfg)
        for sample, lp, cost in zip(samples, scores, costs):
            p = sample['student_lp']
            response_ids = sample['response_ids']
            observed_student = p.gather(1, response_ids[:, None]).mean().item()
            kl = (p.exp()*(p-lp)).sum(-1).mean().item()
            entropy = -(lp.exp()*lp).sum(-1).mean().item()
            observed = lp.gather(1, response_ids[:, None]).mean().item()
            sample['teachers'].append(lp)
            sample['kls'].append(kl)
            sample.setdefault('observed', []).append(observed)
            sample.setdefault('entropies', []).append(entropy)
            sample['observed_student'] = observed_student
            sample['costs'].append(cost)
    for sample in samples:
        sample['routing_costs'] = routing_costs(measured_costs(sample['costs']), cfg)
        sample['similarity'] = distribution_similarity(sample['teachers'])
        sample['contexts'] = paper_context(
            domain_model.predict(sample['row']['messages'], cfg['teachers']),
            sample['observed'], sample['kls'], sample['entropies'], sample['observed_student'],
            sample['similarity'], sample['routing_costs'])
    # Eq.21 defines one team per mini-batch. Average Eq.9's sample contexts,
    # retaining each seven-coordinate sample context in the log for auditability.
    contexts = np.mean([s['contexts'] for s in samples], axis=0)
    kls = np.array([s['kls'] for s in samples])
    similarity = np.mean([s['similarity'] for s in samples], axis=0)
    selection_costs = np.mean([s['routing_costs'] for s in samples], axis=0)
    costs = [{**samples[0]['costs'][i], **{key: sum(s['costs'][i][key] for s in samples)
              for key in ('prefill_seconds', 'load_and_prefill_seconds')}}
             for i in range(len(cfg['teachers']))]
    return samples, contexts, kls, similarity, selection_costs, costs


def make_batch_targets(samples, chosen, weights, cfg):
    """Freeze one common team's targets, then release unselected distributions."""
    for sample in samples:
        student_lp = sample['student_lp']
        selected_lp = torch.stack([sample['teachers'][i] for i in chosen])
        raw_q = mixture_log_probs(selected_lp, weights)
        sample['target'] = mixture_log_probs(selected_lp, weights, student_lp, cfg['epsilon'])
        sample['selected_lp'] = selected_lp
        # Eq.23-24 use pre-update probabilities, never loss reduction or after_lp.
        sample['r_distill'] = (raw_q-student_lp).gather(1, sample['response_ids'][:, None]).mean().item()
        sample['r_stability'] = -(student_lp.exp()*(student_lp-raw_q)).sum(-1).mean().item()
        del sample['teachers'], sample['student_lp']


def backward_batch(student, samples, cfg, pad_token_id=0):
    """Accumulate the mean of per-sequence losses; never step/clear gradients here."""
    if len(samples) != batch_settings(cfg):
        raise ValueError('Expected one complete effective batch')
    size = cfg['micro_batch_size']
    losses = []
    for start in range(0, len(samples), size):
        micro = samples[start:start+size]
        current = response_log_probs_batch(student, micro, cfg['student_device'], pad_token_id)
        terms = [reverse_kl_loss(lp, s['target'], s['selected_lp'], cfg['top_k'])
                 for lp, s in zip(current, micro)]
        loss = torch.stack(terms).sum()/len(samples)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        losses.extend(t.item() for t in terms)
        loss.backward()
        del current, terms, loss
    return float(np.mean(losses)), losses


def update_witness(grads):
    """Return one high-gradient scalar for a constant-memory update check."""
    if not grads:
        raise RuntimeError('No gradients available for the optimizer update witness')
    name, parameter = max(
        grads, key=lambda pair: pair[1].grad.detach().abs().amax().item())
    flat_gradient = parameter.grad.detach().view(-1)
    index = int(flat_gradient.abs().argmax().item())
    before = parameter.detach().view(-1)[index].clone()
    return name, parameter, index, before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--steps', type=int, help='Number of optimizer updates')
    parser.add_argument('--mode', choices=['bandit', 'fixed', 'random', 'uniform'])
    parser.add_argument('--micro-batch-size', type=int, help='Trajectories per student backward microbatch')
    parser.add_argument('--gradient-accumulation-steps', type=int, help='Microbatches per optimizer update')
    parser.add_argument('--save-steps', type=int, help='Save every N optimizer updates; default 50, 0 final-only')
    parser.add_argument('--teacher-device', help='cpu or cuda:N')
    parser.add_argument('--teacher-dtype', choices=['float32', 'bfloat16'])
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key in ('output', 'steps', 'mode', 'teacher_device', 'teacher_dtype', 'micro_batch_size', 'gradient_accumulation_steps', 'save_steps'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if any(t['domain'] == 'agent' for t in cfg['teachers']):
        raise ValueError('Agent teachers are disabled in the four-domain training scope')
    effective_batch = batch_settings(cfg)
    max_empty = cfg.setdefault('max_consecutive_empty_batches', 8)
    if type(max_empty) is not int or max_empty < 1:
        raise ValueError('max_consecutive_empty_batches must be a positive integer')
    context_settings = configure_context(cfg)
    if cfg.get('memory_backend', 'dense') not in ('dense', 'chunked'):
        raise ValueError('Unknown memory_backend')
    if cfg.get('inference_backend', 'transformers') not in ('transformers', 'vllm'):
        raise ValueError('Unknown inference_backend')
    configure_checkpointing(cfg)
    if cfg['steps'] < 1 or not 1 <= cfg['k'] <= len(cfg['teachers']):
        raise ValueError('Invalid steps/team size')
    if cfg['temperature'] != 1.0:
        raise ValueError('Use temperature 1 without top-p/top-k sampling for the on-policy reference')
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    preflight = inspect(cfg)
    reserve_bytes = disk_reserve_bytes(cfg)
    if shutil.disk_usage(storage_parent(output)).free < reserve_bytes:
        raise RuntimeError(f'Insufficient disk for retained checkpoints: need about {reserve_bytes/1024**3:.0f} GiB')
    torch.set_num_threads(cfg['cpu_threads'])
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['student'], local_files_only=True)
    train = read_training_prompts(cfg['train_data'])
    task_weight = cfg['reward'].get('task', 0.0)
    probes = read_jsonl(cfg['probe_data']) if task_weight else []
    if {identity(r['messages']) for r in train} & {identity(r['messages']) for r in probes}:
        raise ValueError('Train/probe prompt leakage')
    buckets = defaultdict(list)
    overlong = 0
    for row in train:
        if row['domain'] not in cfg['domain_filter']:
            continue
        length = len(render_rollout_prompt(tokenizer, row['messages'], cfg, tokenize=True,
                                                  add_generation_prompt=True, return_dict=False))
        if length > cfg['max_prompt_tokens']:
            overlong += 1
            continue
        buckets[row['domain']].append(row)
    if not buckets:
        raise ValueError('No usable training prompts')
    if not cfg['smoke'] and set(buckets) != set(cfg['domain_filter']):
        raise ValueError('A requested domain has no usable examples')
    for rows in buckets.values():
        rng.shuffle(rows)
    if task_weight and 'probe' not in cfg:
        raise ValueError('Nonzero reward.task requires an explicit multi-domain probe configuration')
    probe_evaluator = ProbeEvaluator(probes, tokenizer, cfg) if task_weight else None
    if cfg.get('inference_backend') == 'vllm' and probe_evaluator:
        raise ValueError('vLLM rollout currently requires reward.task=0')
    reward_scope = 'verifier_probe_gain_plus_dense' if probe_evaluator else 'dense_proxy_only'
    domain_model = PromptDomainModel([t['domain'] for t in cfg['teachers']],
        context_settings['hash_bins'], context_settings['smoothing']).fit(train)
    print(json.dumps({'domain_estimator': domain_model.manifest()}), flush=True)
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2))
    domain_model.save(output/'domain_estimator.npz')
    (output/'domain_estimator.json').write_text(json.dumps(domain_model.manifest(), indent=2))
    (output/'preflight.json').write_text(json.dumps(preflight, indent=2))
    if probe_evaluator:
        (output/'probe_manifest.json').write_text(json.dumps(probe_evaluator.manifest, indent=2))
    sources = {str(p): sha256(p) for root in ('bandit_mopd', 'third_party/ifevalg')
               for p in Path(root).rglob('*.py')}
    (output/'provenance.json').write_text(json.dumps({'code_sha256': sources,
        'train_sha256': sha256(cfg['train_data']), 'probe_sha256': sha256(cfg['probe_data']) if task_weight else None,
        'filtered_overlong_prompts': overlong, 'candidate_prefill': 'all_teachers_on_every_trajectory',
        'effective_batch_size': effective_batch, 'step_unit': 'optimizer_update',
        'batch_policy': 'one_domain_one_teacher_team_per_update', 'loss_reduction': 'mean_of_sequence_means',
        'cost_reduction': 'mean_of_per_trajectory_declared_cost',
        'raw_cost_reduction': 'mean_measured_prefill_seconds_per_training_example',
        'context_features': CONTEXT_FEATURES, 'context_definition': context_settings,
        'domain_estimator': domain_model.manifest(),
        'mixture_utility': 'predicted_mean_eq27',
        'diversity_definition': 'sum_unordered_pairs_one_minus_bhattacharyya',
        'reward_scope': reward_scope,
        'bandit_accounting': 'teacher_global'}, indent=2))
    # Retain the project-wide lock so old dual-GPU and new CPU-teacher runs
    # cannot accidentally train overlapping students concurrently.
    import fcntl
    lock = open('.bandit-gpus.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('Loading full student', flush=True)
    student = AutoModelForCausalLM.from_pretrained(cfg['student'], local_files_only=True,
        dtype=torch.bfloat16, device_map=cfg['student_device'], attn_implementation='sdpa')
    student.requires_grad_(True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    # eval() removes dropout; it DOES NOT disable full parameter gradients.
    student.eval()
    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if total != trainable:
        raise RuntimeError('Full parameter training requires every parameter trainable')
    print(json.dumps({'parameters': total, 'trainable_parameters': trainable}), flush=True)
    vllm_rollout = None
    if cfg.get('inference_backend') == 'vllm':
        from .vllm_rollout import VLLMRollout
        vllm_rollout = VLLMRollout(cfg)
    backend = None
    if cfg.get('memory_backend') == 'chunked':
        from .chunked import ResidentTeachers
        backend = ResidentTeachers(cfg)
    optimizer = CPUAdamW(student.parameters(), lr=cfg['lr'])
    domains = sorted(buckets)
    bandit = CombinatorialLinUCB(
        len(cfg['teachers']), len(CONTEXT_FEATURES), cfg['alpha'], cfg['rho'],
        diversity=cfg['diversity'], cost=cfg['selection_cost'])
    positions = defaultdict(int)
    successful = 0
    train_samples = 0
    checkpoint_paths = []
    start_run = time.perf_counter()
    log_file = (output/'metrics.jsonl').open('w', buffering=1)
    attempted_batches = skipped_batches = empty_streak = 0
    while successful < cfg['steps']:
        step = attempted_batches
        attempted_batches += 1
        start = time.perf_counter()
        domain = domains[step % len(domains)]
        print(json.dumps({'event':'step_started','optimizer_step':successful+1,'attempt':attempted_batches,'total_steps':cfg['steps'],
            'domain':domain,'effective_batch_size':effective_batch,
            'timestamp_utc':datetime.now(timezone.utc).isoformat()}),flush=True)
        rows = [buckets[domain][(positions[domain]+i) % len(buckets[domain])]
                for i in range(effective_batch)]
        positions[domain] += effective_batch
        rollout_fn = rollout
        if vllm_rollout:
            vllm_rollout.ensure_weights(student, successful)
            generated = iter(vllm_rollout.generate_batch(rows, tokenizer, cfg, attempted_batches))
            rollout_fn = lambda *_: next(generated)
        samples, contexts, sample_kls, similarity, selection_costs, costs = (
            backend.prepare(student, tokenizer, rows, cfg, rollout_fn, domain_model) if backend else
            prepare_batch(student, tokenizer, rows, cfg, domain_model, rollout_fn))
        # A common team must satisfy the original KL gate for EVERY trajectory.
        eligible = sample_kls.max(axis=0) <= cfg['kl_gate']
        chosen, means, scores = bandit.select(
            contexts, cfg['k'], eligible, similarity, selection_costs)
        eligible_ids = np.flatnonzero(eligible).tolist()
        if cfg['mode'] == 'fixed':
            chosen = [i for i, t in enumerate(cfg['teachers']) if t['domain'] == domain and eligible[i]]
        elif cfg['mode'] == 'random':
            chosen = rng.choice(eligible_ids, min(cfg['k'], len(eligible_ids)), replace=False).tolist()
        elif cfg['mode'] == 'uniform':
            chosen = eligible_ids
        if not chosen:
            skipped_batches += 1
            empty_streak += 1
            exhausted = empty_streak >= max_empty
            event = {'step':step, 'attempt':attempted_batches, 'optimizer_step':successful+1,
                'completed_steps':successful, 'skipped_batches':skipped_batches,
                'consecutive_empty_batches':empty_streak,
                'status':'failed_empty_team_limit' if exhausted else 'skipped_empty_team', 'domain':domain,
                'prompt_ids':[s['row']['id'] for s in samples],
                'response_tokens_per_sample':[len(s['response_ids']) for s in samples],
                'teacher_costs':costs,
                'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'candidate_kl_max':sample_kls.max(axis=0).tolist(),
                'eligible':eligible.tolist(), 'contexts':contexts.tolist(),
                'predicted_utilities':means.tolist(), 'ucb':scores.tolist(),
                'selection_costs':selection_costs.tolist(), 'selection_cost_unit':context_settings['cost'],
                'selection_cost_seconds':np.mean([measured_costs(s['costs']) for s in samples],axis=0).tolist(),
                'initial_marginal_gains':(scores-cfg['selection_cost']*selection_costs).tolist(),
                'reason':('no_kl_eligible_teacher' if not eligible.any() else
                    'negative_initial_marginal_gains' if cfg['mode']=='bandit' else 'empty_team_for_requested_mode'),
                'step_seconds':time.perf_counter()-start}
            log_file.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
            del samples
            gc.collect()
            release_cache(cfg['student_device'])
            if exhausted:
                checkpoint = None
                if successful and (cfg['save_student'] or cfg['save_optimizer']):
                    checkpoint = output/f'checkpoint-{successful}'
                    if not checkpoint.exists():
                        print(f'Saving last valid weights after {successful} updates before stopping',flush=True)
                        checkpoint = save_checkpoint(output,student,tokenizer,optimizer,bandit,cfg,
                            {'optimizer_step':successful,'train_samples_seen':train_samples,
                             'effective_batch_size':effective_batch,'domain_positions':dict(positions),
                             'attempted_batches':attempted_batches,'skipped_batches':skipped_batches,
                             'save_reason':'empty_team_limit'})
                (output/'summary.json').write_text(json.dumps({'status':'failed',
                    'reason':'consecutive_empty_team_limit','steps':successful,'requested_steps':cfg['steps'],
                    'attempted_batches':attempted_batches,'skipped_batches':skipped_batches,
                    'checkpoint':str(checkpoint) if checkpoint else None,'resume_supported':False},indent=2))
                raise RuntimeError(f'No eligible team selected in {max_empty} consecutive batches; stopped without bypassing selection')
            # No student/optimizer/router update; sample the next domain/batch.
            continue
        empty_streak = 0
        weights = torch.as_tensor(sequence_weights(means, chosen, cfg['beta']), dtype=torch.float32)
        if cfg['mode'] != 'bandit':
            weights = torch.ones(len(chosen))/len(chosen)
        if backend:
            backend.targets(student, samples, chosen, weights, cfg)
        else:
            make_batch_targets(samples, chosen, weights, cfg)
        probe_batch = probe_evaluator.batch(step, domain) if probe_evaluator else None
        pre_probe = probe_evaluator.evaluate(student, tokenizer, probe_batch, rollout) if probe_evaluator else None
        if pre_probe:
            (output/f'probe-{step:04d}-before.json').write_text(json.dumps(pre_probe, ensure_ascii=False, indent=2))
        optimizer.zero_grad()
        student.train()
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in student.modules()):
            raise RuntimeError('Nonzero dropout violates the fixed policy scoring assumption')
        if backend:
            from .chunked import student_loss, backward_resident_batch
            sample_losses = backward_resident_batch(student, samples, cfg,
                getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'eos_token_id', None) or 0)
            loss_value = float(np.mean(sample_losses))
        else:
            loss_value, sample_losses = backward_batch(student, samples, cfg,
                getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'eos_token_id', None) or 0)
        grads = [(name, p) for name, p in student.named_parameters() if p.grad is not None]
        missing = [name for name, p in student.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError(f'Parameters without gradients: {missing}')
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        # Verify a high-gradient element changes without cloning/reducing a full tensor.
        witness_name, witness, witness_index, before = update_witness(grads)
        tick = time.perf_counter()
        print(f"step {step}: CPU full-parameter AdamW update", flush=True)
        optimizer.step()
        update_seconds = time.perf_counter()-tick
        # Gradients occupy about one full BF16 model; release them before diagnostics.
        optimizer.zero_grad()
        witness_changed = int(
            witness.detach().view(-1)[witness_index].ne(before).item())
        del before
        student.eval()
        after_losses = []
        with torch.no_grad():
            for sample in samples:
                if backend:
                    after_losses.append(student_loss(student, sample, backend.logit_chunk))
                else:
                    after_lp = log_probs(student, sample['ids'].to(cfg['student_device']), sample['prompt_length']).cpu()
                    after_losses.append(reverse_kl_loss(after_lp, sample['target'], sample['selected_lp'], cfg['top_k']).item())
                    del after_lp
        after_loss = float(np.mean(after_losses))
        post_probe = probe_evaluator.evaluate(student, tokenizer, probe_batch, rollout) if probe_evaluator else None
        task_gain = probe_evaluator.gain(pre_probe, post_probe) if probe_evaluator else None
        r_task = task_gain['normalized_gain'] if task_gain else 0.0
        if post_probe:
            (output/f'probe-{step:04d}-after.json').write_text(json.dumps(post_probe, ensure_ascii=False, indent=2))
        r_distill = float(np.mean([s['r_distill'] for s in samples]))
        r_stability = float(np.mean([s['r_stability'] for s in samples]))
        r_cost = float(selection_costs[chosen].sum())
        r_redundancy = float(sum(similarity[i,j] for i in chosen for j in chosen if i != j)/len(chosen)**2)
        rc = cfg['reward']
        weighted_reward = {'task':rc.get('task',0.0)*r_task, 'distill':rc['distill']*r_distill,
            'stability':rc['stability']*r_stability, 'cost':-rc['cost']*r_cost,
            'redundancy':-rc['redundancy']*r_redundancy}
        reward = sum(weighted_reward.values())
        if cfg['mode'] == 'bandit':
            bandit.update(chosen, contexts, reward, weights.numpy())
        train_samples += len(samples)
        event = {'step': step, 'attempt':attempted_batches, 'skipped_batches':skipped_batches,
            'optimizer_step': successful+1, 'status': 'updated', 'mode': cfg['mode'], 'domain': domain,
            'prompt_id': rows[0]['id'] if len(rows) == 1 else None,
            'prompt_ids': [r['id'] for r in rows], 'effective_batch_size': effective_batch,
            'micro_batch_size': cfg['micro_batch_size'], 'gradient_accumulation_steps': cfg['gradient_accumulation_steps'],
            'student_backward_batch_size': cfg.get('student_backward_batch_size', cfg['micro_batch_size']),
            'train_samples_seen': train_samples,
            'response_tokens': sum(len(s['response_ids']) for s in samples),
            'response_tokens_per_sample': [len(s['response_ids']) for s in samples],
            'truncated': any(s['truncated'] for s in samples),
            'truncated_count': sum(s['truncated'] for s in samples),
            'selected': [cfg['teachers'][i]['name'] for i in chosen], 'weights': weights.tolist(),
            'candidate_kl': sample_kls.mean(axis=0).tolist(), 'candidate_kl_max': sample_kls.max(axis=0).tolist(),
            'context_features': CONTEXT_FEATURES,
            'sample_contexts': [np.asarray(s['contexts']).tolist() for s in samples],
            'contexts': contexts.tolist(), 'predicted_utilities': means.tolist(),
            'selection_costs': selection_costs.tolist(),
            'selection_cost_unit': context_settings['cost'],
            'selection_cost_seconds': np.mean([measured_costs(s['costs']) for s in samples],axis=0).tolist(), 'teacher_similarity': similarity.tolist(),
            'ucb': scores.tolist(), 'loss': loss_value,
            'sample_losses': sample_losses, 'sample_losses_after': after_losses,
            'same_rollout_loss_after': after_loss, 'grad_norm': float(grad_norm),
            'parameters_with_grad': sum(p.numel() for _, p in grads),
            'witness_name': witness_name, 'witness_index': witness_index,
            'witness_changed_elements': witness_changed,
            'reward': reward, 'weighted_reward_terms': weighted_reward,
            'selection_cost_penalty': (cfg['selection_cost']*selection_costs).tolist(),
            'reward_terms': {'task': r_task, 'distill': r_distill,
            'stability': r_stability, 'cost': r_cost, 'redundancy': r_redundancy},
            'bandit_account': bandit.account_summary(),
            'teacher_costs': costs, 'all_teacher_prefill_seconds': sum(c['prefill_seconds'] for c in costs),
            'task_gain': task_gain,
            'probe_seconds': (pre_probe['seconds']+post_probe['seconds']) if probe_evaluator else 0.0,
            'cpu_optimizer_seconds': update_seconds, 'step_seconds': time.perf_counter()-start,
            'cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'gpu_peak_bytes': peak_gpu_memory(cfg)}
        elapsed = time.perf_counter()-start_run
        event.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),learning_rate=cfg['lr'],
            elapsed_seconds=elapsed,eta_seconds=elapsed/(successful+1)*(cfg['steps']-successful-1),
            response_tokens_per_second=event['response_tokens']/event['step_seconds'],
            rollout_seconds=sum(s.get('rollout_seconds',0.0) for s in samples) if backend else None,
            same_rollout_loss_change=after_loss-loss_value,
            memory_peak_scope='process_lifetime')
        log_file.write(json.dumps(event)+'\n'); print(json.dumps(event), flush=True)
        trajectories = [{'prompt_id': s['row']['id'], 'response': s['response'],
            'response_ids': s['response_ids'].tolist(), 'truncated': s['truncated']} for s in samples]
        record = trajectories[0] if len(trajectories) == 1 else {'optimizer_step': successful+1, 'samples': trajectories}
        (output/f'rollout-{step:04d}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        # Do not retain previous targets while generating the next effective batch.
        del samples, sample, grads, witness
        gc.collect()
        release_cache(cfg['student_device'])
        successful += 1
        if should_save(successful, cfg):
            print(f'Saving checkpoint after optimizer update {successful}', flush=True)
            checkpoint = save_checkpoint(output, student, tokenizer, optimizer, bandit, cfg,
                {'optimizer_step': successful, 'train_samples_seen': train_samples,
                 'effective_batch_size': effective_batch, 'domain_positions': dict(positions),
                 'attempted_batches':attempted_batches,'skipped_batches':skipped_batches})
            checkpoint_paths.append(str(checkpoint))
    log_file.close()
    if not successful:
        raise RuntimeError('No student update completed')
    (output/'bandit_state.json').write_text(json.dumps(bandit.state_dict()))
    if checkpoint_paths:
        link_final(output, checkpoint_paths[-1], cfg)
    summary = {'status': 'completed', 'steps': successful, 'parameters': total,
        'attempted_batches':attempted_batches,'skipped_batches':skipped_batches,
        'step_unit': 'optimizer_update', 'effective_batch_size': effective_batch, 'train_samples_seen': train_samples,
        'trainable_parameters': trainable, 'seconds': time.perf_counter()-start_run,
        'smoke': cfg['smoke'], 'benchmark_result': False,
        'reward_scope': reward_scope,
        'checkpoint': str(output/'student') if cfg['save_student'] else None,
        'save_steps': cfg['save_steps'], 'checkpoints': checkpoint_paths,
        'resume_supported': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if vllm_rollout:
        vllm_rollout.close()
    return vllm_rollout is not None


if __name__ == '__main__':
    used_cuda_ipc = main()
    if used_cuda_ipc:
        # PyTorch's CUDA IPC producer destructor crashes during interpreter
        # teardown on the tested torch/vLLM build, after vLLM has shut down.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
