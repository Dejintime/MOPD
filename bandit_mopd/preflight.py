import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import torch
from transformers import AutoTokenizer
from .devices import configure_devices, cuda_devices
from .batching import batch_settings


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def effective_ram_available(host_available):
    """Respect a container's cgroup-v2 limit instead of host RAM alone."""
    root = Path('/sys/fs/cgroup')
    try:
        limit = (root/'memory.max').read_text().strip()
        if limit != 'max':
            current = int((root/'memory.current').read_text().strip())
            # Inactive file cache can be reclaimed under the cgroup limit;
            # otherwise a recently stopped model run falsely blocks restart.
            try:
                stats = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
                file_bytes = int(stats.get('file',stats.get('inactive_file',0)))
                # Include clean, unmapped active file cache left by GPU model
                # loading; it is reclaimable too. Do not count live mappings,
                # shared memory, or pending writes as available working RAM.
                unavailable = sum(int(stats.get(k,0)) for k in
                                  ('file_mapped','shmem','file_dirty','file_writeback'))
                reclaimable = min(current,max(0,file_bytes-unavailable))
            except (OSError,ValueError):
                reclaimable = 0
            return min(host_available, max(0, int(limit)-current+reclaimable))
    except (OSError, ValueError):
        pass
    return host_available


def inspect(config, hash_weights=False):
    effective_batch = batch_settings(config)
    configure_devices(config)
    required = cuda_devices(config)
    if not torch.cuda.is_available() or any(i >= torch.cuda.device_count() for i in required):
        raise RuntimeError(f'Configured CUDA devices unavailable: {required}')
    paths = [config['student']] + [t['path'] for t in config['teachers']]
    models = []
    reference = None
    for path in paths:
        p = Path(path)
        cfg = json.loads((p/'config.json').read_text())
        tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
        signature = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()
        special = (tok.eos_token_id, tok.bos_token_id, tok.chat_template)
        current = (signature, special, cfg['vocab_size'])
        if reference is None:
            reference = current
        if reference != current:
            raise ValueError(f'Token mapping/special tokens/template mismatch: {path}')
        if (p/'model.safetensors.index.json').exists():
            index = json.loads((p/'model.safetensors.index.json').read_text())
            shards = sorted(set(index['weight_map'].values()))
        elif (p/'model.safetensors').is_file():
            shards = ['model.safetensors']
        else:
            raise ValueError(f'Missing safetensors model: {path}')
        if any(not (p/name).is_file() for name in shards):
            raise ValueError(f'Missing model shard: {path}')
        models.append({'path': str(p), 'vocab_sha256': signature,
                       'config_sha256': sha256(p/'config.json'),
                       'weights': {name: {'bytes': (p/name).stat().st_size,
                                         'sha256': sha256(p/name) if hash_weights else None}
                                   for name in shards},
                       'upstream_weight_revision': 'unknown (local downloads)'})
    gpus = [{'index': i, 'name': torch.cuda.get_device_name(i),
             'free_bytes': torch.cuda.mem_get_info(i)[0],
             'total_bytes': torch.cuda.mem_get_info(i)[1]} for i in required]
    for gpu in gpus:
        if gpu['free_bytes'] < 20*1024**3:
            raise RuntimeError(f"Insufficient free memory on cuda:{gpu['index']}; do not displace other jobs")
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = effective_ram_available(int(meminfo['MemAvailable'].split()[0])*1024)
    # CPU teachers coexist with the student's FP32 optimizer states. This is a
    # conservative floor for the current 4B models, not a sequence-length guarantee.
    minimum_ram_gib = 85 if config['teacher_device'] == 'cpu' else 70
    # Candidate distributions live on CPU until one team is selected for the
    # entire update. Reserve additional room for each extra trajectory.
    extra_logits_bytes = ((effective_batch-1) * (len(config['teachers'])+1)
                          * config.get('max_new_tokens', 512) * reference[2] * 4)
    minimum_ram_gib += math.ceil(extra_logits_bytes / 1024**3)
    if config.get('memory_backend') == 'chunked':
        if config['teacher_device'] != 'cpu' or config['teacher_dtype'] not in ('bfloat16','float32'):
            raise ValueError('Resident teachers require CPU BF16 or FP32 placement')
        # Current safetensors are BF16. Count FP32 master/m/v (6x BF16
        # student bytes) plus resident BF16 teachers and 8 GiB working room.
        sizes = [sum(v['bytes'] for v in m['weights'].values()) for m in models]
        teacher_factor = 2 if config['teacher_dtype'] == 'float32' else 1
        # All candidate hidden states coexist until one common batch team is selected.
        student_cfg = json.loads((Path(config['student'])/'config.json').read_text())
        teacher_cfgs = [json.loads((Path(t['path'])/'config.json').read_text()) for t in config['teachers']]
        teacher_bytes = 2*teacher_factor
        cached = effective_batch*config['max_new_tokens']*(2*student_cfg['hidden_size']
            +teacher_bytes*sum(t['hidden_size'] for t in teacher_cfgs))
        total_tokens = config['max_prompt_tokens']+config['max_new_tokens']
        kv = max(2*t['num_hidden_layers']*t['num_key_value_heads']
                 *t.get('head_dim',t['hidden_size']//t['num_attention_heads'])*total_tokens*teacher_bytes
                 for t in teacher_cfgs)
        reserve = config.get('host_memory_reserve_gib',6)
        if not isinstance(reserve,(int,float)) or not math.isfinite(reserve) or reserve<0:
            raise ValueError('host_memory_reserve_gib must be finite and nonnegative')
        # 4 GiB for probability blocks, CPU allocator/library overhead, and prompt data.
        minimum_ram_gib = (6*sizes[0]+teacher_factor*sum(sizes[1:])+cached+kv)/1024**3+4+reserve
        extra_logits_bytes = 0
    if available < minimum_ram_gib*1024**3:
        raise RuntimeError(f'At least {minimum_ram_gib:.1f} GiB available RAM required for this placement')
    return {'models': models, 'gpu': gpus, 'ram_available_bytes': available,
            'minimum_ram_gib': minimum_ram_gib,
            'effective_batch_size': effective_batch,
            'additional_batch_logits_bytes': extra_logits_bytes,
            'placement': {k: config[k] for k in ('student_device', 'teacher_device',
                                                'teacher_dtype', 'cuda_visible_devices')},
            'disk_free_bytes': shutil.disk_usage('.').free, 'python': platform.python_version(),
            'versions': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'accelerate', 'numpy')},
            'training': 'full_parameter_bf16_with_fp32_cpu_adamw'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--hash-weights', action='store_true')
    a = p.parse_args()
    result = inspect(json.loads(Path(a.config).read_text()), a.hash_weights)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=2))
