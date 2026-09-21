import copy
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from bandit_mopd.batching import batch_settings, response_log_probs_batch
from bandit_mopd.train import backward_batch
from bandit_mopd.core import mixture_log_probs, reverse_kl_loss
from bandit_mopd.optimizer import CPUAdamW


def model():
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()


def trajectories():
    return [{'ids': torch.tensor([ids]), 'prompt_length': prompt} for ids, prompt in [
        ([1, 2, 3], 2), ([1, 3, 4, 5, 6], 2),
        ([2, 5, 1, 6, 3, 8], 4), ([3, 1, 2, 7, 9, 4, 6], 3)]]


@pytest.mark.parametrize('key,value', [('micro_batch_size', 0), ('micro_batch_size', True),
    ('gradient_accumulation_steps', -1), ('gradient_accumulation_steps', 1.5)])
def test_invalid_batch_settings(key, value):
    with pytest.raises(ValueError): batch_settings({key: value})


def test_old_configs_default_to_one_sample():
    cfg = {}
    assert batch_settings(cfg) == 1
    assert cfg == {'micro_batch_size': 1, 'gradient_accumulation_steps': 1}


def test_right_padding_and_response_alignment_match_independent_full_forwards():
    m = model(); samples = trajectories()
    actual = response_log_probs_batch(m, samples, 'cpu')
    for result, sample in zip(actual, samples):
        full = m(input_ids=sample['ids']).logits
        expected = full[0, sample['prompt_length']-1:-1].float().log_softmax(-1)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-6)
    # Arbitrary valid padding IDs cannot change non-padding scores.
    other = response_log_probs_batch(m, samples, 'cpu', pad_token_id=16)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(actual, other))


@pytest.mark.parametrize('micro,accumulation', [(1, 4), (2, 2), (4, 1)])
def test_accumulation_matches_full_mean_gradient_and_one_adam_update(micro, accumulation):
    torch.manual_seed(3)
    reference = model(); actual = copy.deepcopy(reference)
    samples = trajectories()
    for sample in samples:
        count = sample['ids'].shape[1]-sample['prompt_length']
        teachers = torch.randn(2, count, 17).log_softmax(-1)
        sample.update(selected_lp=teachers, target=mixture_log_probs(teachers, [.4, .6]))
    terms = []
    for sample in samples:
        # Independent dense forwards; average each response first, then samples.
        lp = reference(sample['ids']).logits[0, sample['prompt_length']-1:-1].log_softmax(-1)
        terms.append(reverse_kl_loss(lp, sample['target'], sample['selected_lp'], 4))
    expected_loss = torch.stack(terms).mean()
    expected_loss.backward()
    cfg = {'micro_batch_size': micro, 'gradient_accumulation_steps': accumulation,
           'student_device': 'cpu', 'top_k': 4}
    got, sample_losses = backward_batch(actual, samples, cfg)
    assert got == pytest.approx(expected_loss.item(), abs=1e-6)
    assert sample_losses == pytest.approx([t.item() for t in terms], abs=1e-6)
    for p, q in zip(actual.parameters(), reference.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=2e-6, rtol=1e-5)
    torch.nn.utils.clip_grad_norm_(actual.parameters(), .1)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
    CPUAdamW(actual.parameters(), lr=1e-3).step()
    CPUAdamW(reference.parameters(), lr=1e-3).step()
    for p, q in zip(actual.parameters(), reference.parameters()):
        assert torch.allclose(p, q, atol=2e-6, rtol=1e-5)
