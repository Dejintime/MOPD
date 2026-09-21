import json
from types import SimpleNamespace

import pytest
import torch

from bandit_mopd.devices import configure_devices, cuda_devices, peak_gpu_memory
from bandit_mopd.train import log_probs, score_teacher, update_witness
from bandit_mopd.core import mixture_log_probs, reverse_kl_loss


def placement(teacher='cpu', student='cuda:0'):
    return {'student_device': student, 'teacher_device': teacher}


@pytest.mark.parametrize('visible,student,expected', [
    (None, 'cuda:0', '0'), ('0,1', 'cuda:0', '0'),
    ('3,5', 'cuda:1', '5'), ('GPU-assigned', 'cuda:0', 'GPU-assigned')])
def test_cpu_teacher_masks_other_gpus(monkeypatch, visible, student, expected):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    cfg = placement(student=student)
    configure_devices(cfg)
    assert cfg['cuda_visible_devices'] == expected
    assert cfg['student_device'] == 'cuda:0'
    assert cfg['teacher_dtype'] == 'float32'
    assert cuda_devices(cfg) == [0]
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated',
                        lambda i: 123 if i == 0 else pytest.fail('Queried GPU1'))
    assert peak_gpu_memory(cfg) == {'0': 123}


@pytest.mark.parametrize('visible', ['', '-1', '0'])
def test_reject_unavailable_student(monkeypatch, visible):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    with pytest.raises(ValueError, match='outside'):
        configure_devices(placement(student='cuda:1'))


def test_reject_late_isolation(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='before CUDA'):
        configure_devices(placement())


def test_dual_gpu_backwards_compatible(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    cfg = placement('cuda:1')
    configure_devices(cfg)
    assert cfg['cuda_visible_devices'] == '2,3'
    assert cfg['teacher_dtype'] == 'bfloat16'
    assert cuda_devices(cfg) == [0, 1]


def test_preflight_queries_only_student_gpu(monkeypatch, tmp_path):
    from bandit_mopd import preflight
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'student' if i == 0 else pytest.fail('GPU1'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info',
                        lambda i: (23*1024**3, 24*1024**3) if i == 0 else pytest.fail('GPU1'))
    (tmp_path/'config.json').write_text(json.dumps({'vocab_size': 2}))
    (tmp_path/'model.safetensors').write_bytes(b'test shard')
    tok = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 1}, eos_token_id=1,
                          bos_token_id=0, chat_template='test')
    monkeypatch.setattr(preflight.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tok)
    original_read = preflight.Path.read_text
    monkeypatch.setattr(preflight.Path, 'read_text', lambda p, *a, **kw:
                        'MemAvailable: 125829120 kB\n' if str(p) == '/proc/meminfo'
                        else original_read(p, *a, **kw))
    cfg = {**placement(), 'student': str(tmp_path), 'teachers': []}
    report = preflight.inspect(cfg)
    assert len(report['gpu']) == 1
    assert report['minimum_ram_gib'] == 85
    assert report['placement']['teacher_device'] == 'cpu'


def test_real_cpu_teacher_scores_and_full_student_gradients(monkeypatch, tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    teacher = Qwen3ForCausalLM(cfg).eval()
    teacher.save_pretrained(tmp_path)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = log_probs(teacher, ids, 3)
    # CPU scoring must work even when every CUDA interaction is forbidden.
    for name in ('synchronize', 'empty_cache', 'device', 'max_memory_allocated', 'mem_get_info'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: pytest.fail('CPU teacher touched CUDA'))
    got, cost = score_teacher({'path': str(tmp_path)}, ids, 3, placement())
    assert got.device.type == 'cpu' and not got.requires_grad
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got.exp().sum(-1), torch.ones(2), atol=1e-6)
    assert cost['device'] == 'cpu' and cost['dtype'] == 'torch.float32'
    student = Qwen3ForCausalLM(cfg).train()
    loss = reverse_kl_loss(log_probs(student, ids, 3), mixture_log_probs(got[None], [1.0]), got[None], 4)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_update_witness_uses_one_high_gradient_element():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    first.grad = torch.tensor([0.1, -3.0, 0.2])
    second.grad = torch.tensor([0.5, 0.4])

    name, parameter, index, before = update_witness(
        [('first', first), ('second', second)])

    assert name == 'first' and parameter is first and index == 1
    assert before.ndim == 0 and before.item() == 2.0
    with torch.no_grad():
        parameter.view(-1)[index].add_(1)
    assert parameter.detach().view(-1)[index].ne(before).item()
