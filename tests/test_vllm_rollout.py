from contextlib import nullcontext

import torch

from bandit_mopd import vllm_rollout


def test_release_ipc_cache_runs_after_synchronization(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(('sync', device)))
    monkeypatch.setattr(vllm_rollout.gc, 'collect', lambda: calls.append(('gc', None)))
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'ipc_collect', lambda: calls.append(('ipc', None)))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append(('empty', None)))

    vllm_rollout.release_ipc_cache('cuda:0')

    assert calls == [('sync', 'cuda:0'), ('gc', None), ('ipc', None), ('empty', None)]
