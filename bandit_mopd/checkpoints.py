"""Atomic periodic snapshots, counted in completed optimizer updates."""
import json
from pathlib import Path
import shutil
import tempfile
import time
import torch


def configure_checkpointing(cfg):
    interval = cfg.setdefault('save_steps', 50)
    if type(interval) is not int or interval < 0:
        raise ValueError('save_steps must be a nonnegative integer (0 means final-only)')
    return interval


def should_save(step, cfg):
    interval = configure_checkpointing(cfg)
    return bool((cfg['save_student'] or cfg['save_optimizer']) and
                (step == cfg['steps'] or (step > 0 and interval and step % interval == 0)))


def disk_reserve_bytes(cfg):
    """Plan for retained checkpoints; final aliases add no duplicate weights."""
    interval = configure_checkpointing(cfg)
    if not (cfg['save_student'] or cfg['save_optimizer']):
        return 0
    count = (cfg['steps']+interval-1)//interval if interval else 1
    # Conservative existing 4B model allowance, now multiplied by snapshot count.
    return count * (65 if cfg['save_optimizer'] else 12) * 1024**3


def storage_parent(path):
    parent = Path(path).absolute().parent
    while not parent.exists():
        parent = parent.parent
    return parent


def save_checkpoint(output, student, tokenizer, optimizer, bandit, cfg, state):
    """Publish a snapshot only after every requested file is successfully closed.

    Optimizer and router state are saved for audit/future loading. This is not a
    resume implementation; RNG/sampling restoration still needs a separate path.
    """
    output = Path(output)
    step = state['optimizer_step']
    destination = output/f'checkpoint-{step}'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {destination}')
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.checkpoint-{step}-', dir=output))
    start = time.perf_counter()
    try:
        (temporary/'config.json').write_text(json.dumps(cfg, indent=2))
        if cfg['save_student']:
            student.save_pretrained(temporary/'student', safe_serialization=True)
            tokenizer.save_pretrained(temporary/'student')
        if cfg['save_optimizer']:
            torch.save(optimizer.state_dict(), temporary/'optimizer.pt')
        (temporary/'bandit_state.json').write_text(json.dumps(bandit.state_dict()))
        (temporary/'trainer_state.json').write_text(json.dumps({**state,
            'status': 'complete', 'step_unit': 'optimizer_update', 'resume_supported': False}, indent=2))
        if destination.exists():
            raise FileExistsError(f'Checkpoint appeared during save: {destination}')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    record = {'optimizer_step': step, 'path': str(destination),
              'seconds': time.perf_counter()-start,
              'bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())}
    # Existing complete checkpoints remain intact if a later save fails.
    latest = output/'.latest_checkpoint.json.tmp'
    latest.write_text(json.dumps(record, indent=2))
    latest.replace(output/'latest_checkpoint.json')
    with (output/'checkpoint_metrics.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    return destination


def link_final(output, checkpoint, cfg):
    """Keep legacy final paths without storing a second 4B model/optimizer copy."""
    output, checkpoint = Path(output), Path(checkpoint)
    for name, enabled in (('student', cfg['save_student']), ('optimizer.pt', cfg['save_optimizer'])):
        if enabled:
            link = output/name
            if link.exists() or link.is_symlink():
                raise FileExistsError(f'Refusing to replace final output: {link}')
            link.symlink_to(Path(checkpoint.name)/name, target_is_directory=(name == 'student'))
