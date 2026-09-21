import copy
import json
from pathlib import Path

import pytest

from bandit_mopd.probes import ProbeEvaluator
from bandit_mopd.verifiers import VerifierError, VerifierSuite, validate_row
from bandit_mopd.verifiers.code import output_matches
from bandit_mopd.verifiers.text import final_response, math, science, instruction_following


def row(domain, suffix=''):
    return {'id': domain+suffix, 'domain': domain,
        'messages': [{'role': 'user', 'content': 'Question '+domain+suffix}],
        'answer': 'A' if domain == 'science' else '1/2',
        'metadata': {'template_metadata': {'output_regex': r'Answer\s*:\s*([A-J])\s*'},
                     'instruction_id_list': ['keywords:existence', 'punctuation:no_comma'],
                     'kwargs': [{'keywords': ['apple']}, None],
                     'verifier_metadata': {'unit_tests': {'inputs': ['2 3\n', '9 1\n'], 'outputs': ['5\n', '10\n']}}}}


@pytest.mark.parametrize('response,expected', [
    ('Answer: A', 1), ('Answer: B', 0), ('Answer: A/B', 0),
    ('Answer: Apple', 0), ('Answer: A or B', 0),
    ('<think>Answer: A</think>Answer: B', 0), ('<think>Answer: A', 0),
    ('Answer: B\nAnswer: A', 1), (r'\boxed{A}', 1)])
def test_science_final_answer(response, expected):
    assert science(row('science'), response)['score'] == expected


@pytest.mark.parametrize('response,expected', [
    (r'Answer: 0.5', 1), (r'\boxed{\frac{1}{2}}', 1),
    (r'\boxed{\frac{2}{3}}', 0), ('We saw 1/2 earlier.\nAnswer: 7', 0),
    ('<think>Answer: 1/2</think>Answer: 3', 0)])
def test_math_equivalence_and_final_only(response, expected):
    assert math(row('math'), response)['score'] == expected


def test_if_checks_all_constraints_and_thinking_is_not_answer():
    r = row('if')
    assert instruction_following(r, 'apple')['score'] == 1
    result = instruction_following(r, 'apple, pear')
    assert result['score'] == 0 and result['instruction_passed'] == [True, False]
    assert instruction_following(r, '<think>apple</think>pear')['score'] == 0
    assert final_response('<think>unfinished') == ''


def test_invalid_metadata_is_error_not_zero():
    r = row('if'); r['metadata']['kwargs'] = []
    with pytest.raises(VerifierError): validate_row(r)
    r = row('if'); r['metadata']['instruction_id_list'][0] = 'invented:check'
    with pytest.raises(VerifierError): validate_row(r)
    r = row('code'); r['metadata']['verifier_metadata']['unit_tests']['outputs'] = []
    with pytest.raises(VerifierError): validate_row(r)
    with pytest.raises(VerifierError): validate_row(row('agent'))


def test_code_output_comparison():
    assert output_matches('5  \n10\n', '5\n10')
    assert output_matches('0.3333333333', '0.333333')
    assert not output_matches('100000000000000001', '100000000000000002')
    assert not output_matches('nan', '1')
    assert not output_matches('YES', 'NO')


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['return_dict'] is False
        return [1, 2, 3]

    def save_pretrained(self, path):
        (path/'tokenizer.json').write_text('{}')


class FakeSuite:
    def check_ready(self, domains): pass
    def score(self, r, response):
        return {'score': float(response == 'correct')}


def config():
    return {'domain_filter': ['math', 'code', 'science', 'if'], 'seed': 42,
            'max_prompt_tokens': 1024, 'max_new_tokens': 32,
            'probe': {'scope': 'all_domains', 'samples_per_domain': 1, 'gain_scale': 1., 'verifiers': {}}}


def test_paired_macro_gain_keeps_negative_transfer_and_metadata_private():
    cfg = config()
    probes = [row(d, str(i)) for d in cfg['domain_filter'] for i in range(3)]
    evaluator = ProbeEvaluator(probes, Tokenizer(), cfg, suite=FakeSuite())
    batch = evaluator.batch(0, 'math')
    assert batch == evaluator.batch(0, 'science')
    assert set(r['id'] for r in batch).isdisjoint(r['id'] for r in evaluator.batch(1, 'math'))
    def generate(model, tokenizer, r, cfg, greedy):
        assert set(r) == {'id', 'domain', 'messages'}
        assert greedy
        return None, None, 'correct' if r['domain'] in model else 'wrong', False
    before = evaluator.evaluate({'math', 'science'}, Tokenizer(), batch, generate)
    after = evaluator.evaluate({'code'}, Tokenizer(), batch, generate)
    gain = evaluator.gain(before, after)
    assert gain['raw_gain'] == -.25 and gain['normalized_gain'] == -.25
    assert gain['domain_gains'] == {'code': 1., 'if': 0., 'math': -1., 'science': -1.}
    after['records'].reverse()
    with pytest.raises(VerifierError, match='identical'): evaluator.gain(before, after)


def test_probe_requires_each_domain_and_rejects_final_tests():
    with pytest.raises(VerifierError, match='usable'):
        ProbeEvaluator([row('science')], Tokenizer(), config(), suite=FakeSuite())
    r = {**row('science'), 'usage': 'final_evaluation_only'}
    with pytest.raises(VerifierError, match='benchmarks'):
        ProbeEvaluator([r], Tokenizer(), config(), suite=FakeSuite())


def test_real_text_worker():
    assert VerifierSuite({}).score(row('math'), 'Answer: 0.5')['score'] == 1
    assert VerifierSuite({}).score(row('if'), 'apple')['score'] == 1


@pytest.mark.parametrize('micro,accumulation,steps,save_interval',
                         [(1, 1, 1, 0), (1, 3, 2, 0), (2, 2, 2, 0), (1, 2, 3, 2)])
def test_full_training_step_includes_task_gain(monkeypatch, tmp_path, micro, accumulation, steps, save_interval):
    import sys
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from bandit_mopd import train, probes
    model_cfg = Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    for name in ('student', 'teacher'):
        Qwen3ForCausalLM(model_cfg).save_pretrained(tmp_path/name)
    cfg = json.loads(Path('configs/experiments/bandit_full_cpu_teacher_smoke.json').read_text())
    cfg.update(memory_backend='dense', student=str(tmp_path/'student'), student_device='cpu', teachers=[
        {'name': 'T', 'domain': 'math', 'path': str(tmp_path/'teacher')}], steps=steps, k=1,
        micro_batch_size=micro, gradient_accumulation_steps=accumulation,
        output=str(tmp_path/'result'), save_student=bool(save_interval), save_optimizer=bool(save_interval),
        save_steps=save_interval,
        domain_filter=config()['domain_filter'], probe=config()['probe'], kl_gate=100., cpu_threads=1)
    cfg['reward']['task'] = 2.
    for name in ('train', 'probe'):
        p = tmp_path/(name+'.jsonl')
        p.write_text(''.join(json.dumps(row(d, name+str(i)))+'\n' for d in cfg['domain_filter']
                             for i in range(micro*accumulation)))
        cfg[name+'_data'] = str(p)
    p = tmp_path/'config.json'; p.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(p)])
    monkeypatch.setattr(train, 'inspect', lambda cfg: {})
    # Tiny checkpoints do not need production 4B model disk reservations.
    from types import SimpleNamespace
    monkeypatch.setattr(train.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1000*1024**3))
    monkeypatch.setattr(train.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(probes, 'VerifierSuite', lambda cfg: FakeSuite())
    monkeypatch.setattr(train, 'peak_gpu_memory', lambda cfg: {})
    updated = False
    counts = {'optimizer': 0, 'bandit': 0, 'probe': 0, 'training_rollouts': 0, 'teacher_loads': 0}
    original_score = train.score_teacher_batch
    def score_batch(spec, samples, cfg):
        counts['teacher_loads'] += 1
        assert len(samples) == micro*accumulation
        return original_score(spec, samples, cfg)
    monkeypatch.setattr(train, 'score_teacher_batch', score_batch)
    original_bandit_update = train.CombinatorialLinUCB.update
    def bandit_update(bandit, *args):
        counts['bandit'] += 1
        return original_bandit_update(bandit, *args)
    monkeypatch.setattr(train.CombinatorialLinUCB, 'update', bandit_update)
    original = train.CPUAdamW.step
    def update(optimizer):
        nonlocal updated
        original(optimizer)
        counts['optimizer'] += 1
        updated = True
    monkeypatch.setattr(train.CPUAdamW, 'step', update)
    def generate(*args, **kwargs):
        if kwargs.get('greedy'):
            counts['probe'] += 1
        else:
            assert counts['optimizer'] == counts['training_rollouts']//(micro*accumulation)
            counts['training_rollouts'] += 1
        return torch.tensor([[1, 2, 3, 4, 5]]), 3, 'correct' if updated else 'wrong', False
    monkeypatch.setattr(train, 'rollout', generate)
    train.main()
    events = [json.loads(line) for line in (tmp_path/'result/metrics.jsonl').read_text().splitlines()]
    assert counts == {'optimizer': steps, 'bandit': steps, 'probe': 8*steps,
                      'training_rollouts': steps*micro*accumulation, 'teacher_loads': steps}
    assert len(events) == steps
    assert [e['train_samples_seen'] for e in events] == [(i+1)*micro*accumulation for i in range(steps)]
    assert all(len(set(e['prompt_ids'])) == micro*accumulation for e in events)
    event = events[0]
    assert event['task_gain']['normalized_gain'] == 1.
    terms = event['reward_terms']; rc = cfg['reward']
    expected = rc['task'] + rc['distill']*terms['distill'] + rc['stability']*terms['stability'] - rc['cost']*terms['cost'] - rc['redundancy']*terms['redundancy']
    assert event['reward'] == pytest.approx(expected)
    expected_credit = 0.
    for e in events:
        expected_credit = cfg['rho']*expected_credit + e['reward']
    state = json.loads((tmp_path/'result/bandit_state.json').read_text())
    assert state['b'][0][0] == pytest.approx(expected_credit)
    summary = json.loads((tmp_path/'result/summary.json').read_text())
    assert summary['train_samples_seen'] == steps*micro*accumulation
    if save_interval:
        root = tmp_path/'result'
        assert sorted(p.name for p in root.glob('checkpoint-*')) == ['checkpoint-2', 'checkpoint-3']
        for step in (2, 3):
            saved = root/f'checkpoint-{step}'
            state = json.loads((saved/'trainer_state.json').read_text())
            assert state['optimizer_step'] == step
            assert state['train_samples_seen'] == step*micro*accumulation
            optimizer_state = torch.load(saved/'optimizer.pt', weights_only=True)
            assert all(s['step'] == step for s in optimizer_state['state'].values())
            restored = Qwen3ForCausalLM.from_pretrained(saved/'student', local_files_only=True)
            assert all(torch.isfinite(p).all() for p in restored.parameters())
        assert (root/'student').resolve() == root/'checkpoint-3/student'
        assert len(summary['checkpoints']) == 2
    before = json.loads((tmp_path/'result/probe-0000-before.json').read_text())
    after = json.loads((tmp_path/'result/probe-0000-after.json').read_text())
    assert [r['id'] for r in before['records']] == [r['id'] for r in after['records']]
