import numpy as np
import pytest
import torch
from bandit_mopd.core import CombinatorialLinUCB, mixture_log_probs, reverse_kl_loss
from bandit_mopd.optimizer import CPUAdamW
from bandit_mopd.data import normalize, identity, prepare


def test_mixture_is_arithmetic_probability_not_mean_logprob():
    t = torch.tensor([[[.8,.2]], [[.2,.8]]]).log()
    assert torch.allclose(mixture_log_probs(t, [.25,.75]).exp(), torch.tensor([[.35,.65]]))


def test_union_uses_cross_teacher_mass_and_counts_duplicates_once():
    teachers = torch.tensor([[[.6,.3,.1]], [[.2,.7,.1]]]).log()
    p = torch.tensor([[.4,.5,.1]]).log().requires_grad_()
    q = mixture_log_probs(teachers, [.5,.5])
    loss = reverse_kl_loss(p, q, teachers, top_k=1)
    assert abs(loss.item()) < 1e-6  # p equals q on the full union
    logits = torch.tensor([[.2,.7,.1]]).log().requires_grad_()
    plog = logits.log_softmax(-1)
    got = reverse_kl_loss(plog, q, teachers, top_k=1)
    expected = sum(plog[0,j].exp()*(plog[0,j]-q[0,j])-plog[0,j].exp()+q[0,j].exp() for j in (0,1))
    assert torch.allclose(got, expected)
    assert torch.allclose(torch.autograd.grad(got, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0])


def test_equal_distribution_has_zero_topk_gradient():
    logits = torch.tensor([[.4,-.5,.3]], requires_grad=True)
    lp = logits.log_softmax(-1)
    loss = reverse_kl_loss(lp, lp.detach(), lp.detach()[None], top_k=1)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-7)


def test_full_k_equals_full_reverse_kl():
    torch.manual_seed(2)
    teachers = torch.randn(2,3,7).log_softmax(-1)
    p = torch.randn(3,7).log_softmax(-1)
    q = mixture_log_probs(teachers, [.3,.7])
    assert torch.allclose(reverse_kl_loss(p,q,teachers,7), reverse_kl_loss(p,q,teachers), atol=1e-6)


def test_regularized_target_is_normalized_and_detached():
    p = torch.randn(2,5, requires_grad=True).log_softmax(-1)
    t = torch.randn(2,2,5).log_softmax(-1)
    q = mixture_log_probs(t, [.5,.5], p, .05)
    assert not q.requires_grad
    assert torch.allclose(q.exp().sum(-1), torch.ones(2))


def test_bandit_gate_and_weighted_credit_equations():
    b = CombinatorialLinUCB(3,2,rho=.9)
    x = np.array([[1.,0.],[1.,1.],[0.,1.]])
    chosen,_,_ = b.select(x,2,[True,False,True],np.eye(3),np.ones(3))
    assert set(chosen)=={0,2}
    b.update(chosen,x,2.,[.25,.75])
    assert np.allclose(b.A[chosen[0]], .9*np.eye(2)+np.outer(x[chosen[0]],x[chosen[0]]))
    assert np.allclose(b.b[chosen[0]], .5*x[chosen[0]])
    assert b.select(x,2,[False]*3,np.eye(3),np.ones(3))[0]==[]


def test_bandit_keeps_independent_teacher_accounts_per_domain():
    bandit = CombinatorialLinUCB(2, 2, rho=.9, alpha=0, domains=['code', 'math'])
    contexts = np.array([[1., 0.], [0., 1.]])
    initial_code = bandit.scores(contexts, 'code')
    bandit.update([0, 1], contexts, 2., [.25, .75], 'math')

    assert np.array_equal(bandit.scores(contexts, 'code')[0], initial_code[0])
    assert np.allclose(bandit.A[1, 0], .9*np.eye(2)+np.outer(contexts[0], contexts[0]))
    assert np.allclose(bandit.b[1, 0], .5*contexts[0])
    assert bandit.account_summary('math') == {
        'domain': 'math', 'update_counts': [1, 1], 'reward_credit': [.5, 1.5]}
    assert bandit.account_summary('code') == {
        'domain': 'code', 'update_counts': [0, 0], 'reward_credit': [0., 0.]}
    state = bandit.state_dict()
    assert state['accounting'] == 'teacher_by_domain'
    assert state['domains'] == ['code', 'math']
    with pytest.raises(ValueError, match='Unknown bandit domain'):
        bandit.scores(contexts, 'science')


def test_cpu_adamw_matches_pytorch_multiple_steps():
    torch.manual_seed(42)
    a = torch.nn.Parameter(torch.randn(7,9))
    b = torch.nn.Parameter(a.detach().clone())
    ours = CPUAdamW([a],lr=.01,weight_decay=.1,chunk_size=13)
    reference = torch.optim.AdamW([b],lr=.01,weight_decay=.1,foreach=False)
    for _ in range(4):
        grad=torch.randn_like(a)
        a.grad=grad.clone(); b.grad=grad.clone()
        ours.step(); reference.step()
        assert torch.allclose(a,b,atol=2e-7,rtol=1e-6)
        ours.zero_grad(); reference.zero_grad()


def test_teacher_has_no_gradient_but_student_does():
    s=torch.randn(2,7,requires_grad=True)
    t=torch.randn(2,2,7,requires_grad=True)
    tl=t.log_softmax(-1)
    loss=reverse_kl_loss(s.log_softmax(-1),mixture_log_probs(tl,[.5,.5]),tl,2)
    loss.backward()
    assert s.grad.abs().sum()>0 and t.grad is None


def test_placeholder_rejected_and_system_message_preserved():
    row={'dataset':'nano_v3_sft_profiled_dapo17k','_hf_placeholder':{'row':1}}
    assert normalize(row,0)[1]=='unrestored_math_placeholder'
    row={'dataset':'nano_v3_sft_profiled_stem_mcqa','responses_create_params':{
        'input':[{'role':'system','content':'System'},{'role':'user','content':'Question'}]}}
    result,_=normalize(row,0)
    assert len(result['messages'])==2
    assert identity(result['messages']) == identity([{'role':'system','content':' system '},{'role':'user','content':'QUESTION'}])


def test_split_deduplication(tmp_path):
    import json
    rows=[{'dataset':'nano_v3_sft_profiled_stem_mcqa','responses_create_params':{
        'input':[{'role':'user','content':str(i)}]}} for i in range(15)]
    p=tmp_path/'input.jsonl'; p.write_text('\n'.join(json.dumps(x) for x in rows+rows))
    report=prepare(p,tmp_path/'out')
    assert report['rejected']['duplicate_prompt']==15
    sets=[{json.loads(line)['id'] for line in (tmp_path/'out'/f'{s}.jsonl').read_text().splitlines()} for s in ('train','probe','dev')]
    assert len(set.union(*sets))==15
    assert not (sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])


def test_response_logit_alignment_matches_full_sequence():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from bandit_mopd.train import log_probs
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=17,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    ids=torch.tensor([[1,2,3,4,5]])
    got=log_probs(model,ids,3)
    expected=model(ids).logits[0,2:4].float().log_softmax(-1)
    assert got.shape==(2,17) and torch.allclose(got,expected,atol=1e-6)


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    import json
    from bandit_mopd.data import read_jsonl
    p=tmp_path/'data.jsonl'
    row={'text':'line one\u2028line two\u2029line three'}
    p.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
    assert read_jsonl(p)==[row]
