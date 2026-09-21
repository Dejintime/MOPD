"""Bounded replay capacity checks, never a formal on-policy experiment.

Micro/effective phases use synthetic compact targets with real resident model
weights and a real 16K response replay (prompt padded to the configured cap).
The end-to-end phase computes real teacher statistics, routing, and targets.
"""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import resource
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bandit_mopd.chunked import ResidentTeachers, backward_resident_batch
from bandit_mopd.context import PromptDomainModel, CONTEXT_FEATURES, configure_context
from bandit_mopd.core import CombinatorialLinUCB, sequence_weights
from bandit_mopd.data import read_training_prompts
from bandit_mopd.optimizer import CPUAdamW
from bandit_mopd.preflight import effective_ram_available


def host_available():
    values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return effective_ram_available(int(values['MemAvailable'].split()[0])*1024)


def main(argv=None, resident=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--replay',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['micro','effective','end-to-end'],default='micro')
    parser.add_argument('--micro',type=int,default=1)
    parser.add_argument('--accumulation',type=int,default=1)
    parser.add_argument('--vllm',action='store_true',help='Keep colocated vLLM engine resident during the test')
    parser.add_argument('--vllm-warmup',action='store_true',
                        help='Generate full-budget rollouts before the backward test')
    args=parser.parse_args(argv)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text())
    cfg.update(output=str(output),micro_batch_size=args.micro,gradient_accumulation_steps=args.accumulation,
               save_student=False,save_optimizer=False)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    if resident is not None and 'lock' in resident:
        lock=resident['lock']
    else:
        lock=open('.bandit-gpus.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resident is not None: resident['lock']=lock
    stop=threading.Event()
    def monitor():
        with (output/'host-memory.jsonl').open('w',buffering=1) as log:
            while not stop.is_set():
                available=host_available()
                log.write(json.dumps({'time':time.time(),'available_bytes':available})+'\n')
                if available<4*1024**3:
                    (output/'summary.json').write_text(json.dumps({'status':'host_reserve_reached',
                        'minimum_available_gib':4,'phase':args.phase,'micro':args.micro,'accumulation':args.accumulation}))
                    os._exit(77)
                stop.wait(1)
    threading.Thread(target=monitor,daemon=True).start()
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    result=dict(status='running',phase=args.phase,purpose='resource_capacity_only_not_formal_training',
                micro_batch_size=args.micro,gradient_accumulation_steps=args.accumulation,
                effective_batch_size=args.micro*args.accumulation,
                residency='reused_across_capacity_cases' if resident is not None else 'fresh_process')
    started=time.perf_counter()
    vllm=None
    try:
        if resident is not None and 'student' in resident:
            student,backend,optimizer=(resident[k] for k in ('student','backend','optimizer'))
            optimizer.zero_grad();student.eval();gc.collect();torch.cuda.empty_cache()
            print('Reusing all resident teachers, student, and initialized optimizer states',flush=True)
        else:
            print('Loading student and four resident CPU teachers',flush=True)
            student=AutoModelForCausalLM.from_pretrained(cfg['student'],local_files_only=True,
                dtype=torch.bfloat16,device_map=cfg['student_device'],attn_implementation='sdpa')
            student.requires_grad_(True)
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            student.config.use_cache=False;student.eval()
            if args.vllm:
                from bandit_mopd.vllm_rollout import VLLMRollout
                vllm=VLLMRollout(cfg)
                vllm.ensure_weights(student,0)
            backend=ResidentTeachers(cfg)
            assert len(backend.models)==4
            assert all(p.device.type=='cpu' for m in backend.models for p in m.parameters())
            optimizer=CPUAdamW(student.parameters(),lr=cfg['lr'])
            print('Allocating all FP32 Adam master weights and moments before the test',flush=True)
            optimizer.initialize_state()
            if resident is not None:
                resident.update(student=student,backend=backend,optimizer=optimizer)
        result['optimizer_bytes']=sum(t.numel()*t.element_size() for state in optimizer.state.values()
                                      for t in state.values() if isinstance(t,torch.Tensor))
        result['teacher_weight_bytes']=sum(p.numel()*p.element_size() for m in backend.models for p in m.parameters())
        result['teacher_dtype']=cfg['teacher_dtype']
        source=json.loads(Path(args.replay).read_text())
        original_prompt=source['ids'][0][:source['prompt_length']]
        response=source['ids'][0][source['prompt_length']:]
        assert len(response)==cfg['max_new_tokens']
        prompt=(original_prompt*math.ceil(cfg['max_prompt_tokens']/len(original_prompt)))[:cfg['max_prompt_tokens']]
        ids=torch.tensor([prompt+response],dtype=torch.long)
        total=args.micro*args.accumulation
        result.update(prompt_tokens=len(prompt),response_tokens=len(response),total_tokens=ids.numel(),
                      prompt_note='real replay prompt repeated to the configured maximum for capacity stress')
        if args.vllm_warmup:
            if vllm is None:
                raise ValueError('--vllm-warmup requires --vllm')
            from vllm import SamplingParams
            print(f'vLLM full-budget warmup: {args.micro} sequences x {cfg["max_new_tokens"]} tokens', flush=True)
            outputs=vllm.llm.generate(
                [{'prompt_token_ids': prompt} for _ in range(args.micro)],
                SamplingParams(max_tokens=cfg['max_new_tokens'], temperature=1.0,
                               top_p=1.0, top_k=-1, min_p=0.0, ignore_eos=True),
                use_tqdm=False)
            if len(outputs) != args.micro or any(len(x.outputs[0].token_ids) != cfg['max_new_tokens'] for x in outputs):
                raise RuntimeError('vLLM warmup did not reach the configured response budget')
            result['vllm_full_budget_warmup'] = True
            del outputs
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        if args.phase=='end-to-end':
            settings=configure_context(cfg)
            rows=read_training_prompts(cfg['train_data'])
            domain_model=PromptDomainModel([t['domain'] for t in cfg['teachers']],settings['hash_bins'],settings['smoothing']).fit(rows)
            selected_rows=[r for r in rows if r['domain']=='math'][:total]
            tokenizer=AutoTokenizer.from_pretrained(cfg['student'],local_files_only=True)
            replay=lambda *a:(ids,len(prompt),source['response'],True)
            samples,contexts,kls,similarity,costs,timing=backend.prepare(student,tokenizer,selected_rows,cfg,replay,domain_model)
            bandit=CombinatorialLinUCB(4,len(CONTEXT_FEATURES),cfg['alpha'],cfg['rho'],
                                      diversity=cfg['diversity'],cost=cfg['selection_cost'])
            chosen,means,scores=bandit.select(contexts,cfg['k'],kls.max(0)<=cfg['kl_gate'],similarity,costs)
            if len(chosen)!=2: raise RuntimeError(f'Capacity check requires two selected teachers: {chosen}')
            weights=sequence_weights(means,chosen,cfg['beta'])
            backend.targets(student,samples,chosen,weights,cfg)
            result.update(selected=chosen,weights=weights.tolist(),teacher_costs=timing,
                          contexts=contexts.tolist(),target_source='actual_teacher_mixture')
        else:
            # A genuine 128-token union footprint. Targets are synthetic only in this phase.
            union=torch.arange(128).expand(len(response),128).clone()
            template=dict(ids=ids,prompt_length=len(prompt),response_ids=ids[0,len(prompt):],
                union_ids=union,union_valid=torch.ones_like(union,dtype=torch.bool),
                target=torch.full(union.shape,-math.log(128),dtype=torch.float32))
            if args.phase=='effective':
                # Model the pre-selection cache peak for every distinct trajectory.
                # Touch all pages; reserve one actual-size teacher KV cache plus temporary logits.
                hidden=student.config.hidden_size
                teacher=backend.models[0].config
                element_bytes=2 if cfg['teacher_dtype']=='bfloat16' else 4
                kv_bytes=2*teacher.num_hidden_layers*teacher.num_key_value_heads*teacher.head_dim*ids.numel()*element_bytes
                reserve=kv_bytes+2*1024**3
                cache_bytes=total*len(response)*hidden*(2+element_bytes*4)
                if host_available()<reserve+cache_bytes+6*1024**3:
                    raise MemoryError('Insufficient host reserve for all candidate caches and teacher KV workspace')
                scratch=torch.zeros(reserve,dtype=torch.uint8)
                teacher_dtype=torch.bfloat16 if cfg['teacher_dtype']=='bfloat16' else torch.float32
                caches=[torch.zeros(total,len(response),hidden,dtype=dtype)
                        for dtype in [torch.bfloat16]+[teacher_dtype]*4]
                result['cache_stress_bytes']=cache_bytes
                result['teacher_workspace_reserve_bytes']=reserve
                result['cache_peak_host_available_bytes']=host_available()
                del scratch,caches;gc.collect()
            samples=[{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in template.items()} for _ in range(total)]
            result['target_source']='synthetic_compact_targets_for_capacity_only'
        print(f"Full {cfg['max_new_tokens']}-token backward: micro={args.micro}, effective={total}",flush=True)
        optimizer.zero_grad();student.train()
        losses=backward_resident_batch(student,samples,cfg,pad_token_id=0)
        assert all(p.grad is not None for p in student.parameters())
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
        name,witness=max(student.named_parameters(),key=lambda item:item[1].grad.float().abs().max().item())
        before=witness.detach().clone()
        print('CPU Adam update',flush=True)
        tick=time.perf_counter();optimizer.step()
        result['optimizer_seconds']=time.perf_counter()-tick
        result['witness_changed_elements']=int((witness.detach()!=before).sum().item())
        assert result['witness_changed_elements']>0
        result.update(status='passed',losses=losses,grad_norm=float(norm),
            parameters_with_grad=sum(p.numel() for p in student.parameters() if p.grad is not None))
        if args.phase=='end-to-end':
            distill=float(np_mean([s['r_distill'] for s in samples]))
            stability=float(np_mean([s['r_stability'] for s in samples]))
            redundancy=sum(similarity[i,j] for i in chosen for j in chosen if i!=j)/len(chosen)**2
            rc=cfg['reward'];reward=rc['distill']*distill+rc['stability']*stability-rc['cost']*costs[chosen].sum()-rc['redundancy']*redundancy
            bandit.update(chosen,contexts,float(reward),weights)
            result['dense_reward']=float(reward);result['bandit_updated']=True
        del before
    except torch.OutOfMemoryError as error:
        result.update(status='cuda_oom',error=str(error))
    except MemoryError as error:
        result.update(status='host_reserve_insufficient',error=str(error))
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,cpu_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.cuda.is_initialized():
            result.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (output/'summary.json').write_text(json.dumps(result,indent=2))
        stop.set()
        print(json.dumps(result),flush=True)
        if vllm is not None:
            vllm.close()
    return 0 if result['status']=='passed' else 42


def np_mean(values):
    return sum(values)/len(values)


if __name__=='__main__':
    code=main()
    if '--vllm' in __import__('sys').argv:
        # CUDA IPC teardown in this torch/vLLM build crashes after clean shutdown.
        __import__('sys').stdout.flush()
        __import__('os')._exit(code)
    raise SystemExit(code)
