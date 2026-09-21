import copy

import numpy as np
import pytest
import torch

from bandit_mopd.context import (CONTEXT_FEATURES, PromptDomainModel, configure_context,
                               paper_context, measured_costs, routing_costs)
from bandit_mopd.core import (CombinatorialLinUCB, distribution_similarity,
                            sequence_weights, team_objective)


def row(domain, text):
    return dict(domain=domain, messages=[dict(role='user', content=text)])


def test_domain_estimator_learns_prompt_probability_without_reading_current_label(tmp_path):
    training = [row('math', 'equation integral algebra'), row('math', 'solve algebra'),
                row('code', 'python function compiler'), row('code', 'compile python')]
    model = PromptDomainModel(['math', 'code', 'agent'], hash_bins=256).fit(training)
    teachers = [dict(domain=d) for d in ('math', 'code', 'agent')]
    prompt = row('code', 'integral algebra')  # Deliberately misleading label.
    actual = model.predict(prompt['messages'], teachers)
    assert actual[0] > actual[1] > 0
    assert actual[2] == 0
    assert actual.sum() == pytest.approx(1)
    prompt['domain'] = 'agent'
    prompt['answer'] = 'python compiler'
    assert np.array_equal(actual, model.predict(prompt['messages'], teachers))
    assert model.manifest()['missing_domains'] == ['agent']
    assert model.predict([], teachers).tolist() == [.5, .5, 0.]
    model.save(tmp_path/'model.npz')
    with np.load(tmp_path/'model.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['log_likelihood'], model.log_likelihood)
    with pytest.raises(ValueError, match='Final evaluation'):
        model.fit([{**training[0], 'usage': 'final_evaluation_only'}])


def test_eq9_exact_order_units_advantage_and_other_teacher_agreement():
    sim = np.array([[1, .8, .4], [.8, 1, .6], [.4, .6, 1]])
    actual = paper_context([.2,.5,.3], [-2,-3,-4], [.1,.2,.3], [1,2,3], -3.5,
                           sim, [2,4,8])
    assert len(CONTEXT_FEATURES) == 7
    expected = [[.2,-2,.1,1,1.5,.6,2], [.5,-3,.2,2,.5,.7,4], [.3,-4,.3,3,-.5,.5,8]]
    assert np.allclose(actual, expected)
    singleton = paper_context([1],[-2],[0],[1],-2,[[1]],[3])
    assert singleton[0,5] == 0
    assert measured_costs([dict(prefill_seconds=2,load_and_prefill_seconds=100)]) == [2]
    with pytest.raises(ValueError):
        paper_context([1],[-2],[0],[1],-2,[[1]],[-1])
    with pytest.raises(ValueError):
        configure_context({'context': {'schema':'legacy_nine_dimensions'}})


def test_eq28_29_greedy_matches_independent_marginal_objective_and_uses_cost():
    bandit = CombinatorialLinUCB(3, 7, alpha=.5, diversity=.1, cost=.2)
    contexts = np.array([[.5,-2,.1,1,1,.7,1], [.3,-1,.2,2,2,.6,3], [.2,-3,.3,1,0,.5,2]])
    bandit.A = np.asarray([np.diag(np.arange(1,8)+i) for i in range(3)], dtype=float)
    bandit.b = np.arange(21).reshape(3,7)/20
    sim = np.array([[1,.9,.2],[.9,1,.4],[.2,.4,1]])
    costs = np.array([1,3,2])
    chosen, means, scores = bandit.select(contexts,2,[True,False,True],sim,costs)
    expected_means = np.array([x @ np.linalg.inv(a) @ b for x,a,b in zip(contexts,bandit.A,bandit.b)])
    expected_scores = expected_means + .5*np.sqrt([x @ np.linalg.inv(a) @ x for x,a in zip(contexts,bandit.A)])
    assert np.allclose(means,expected_means)
    assert np.allclose(scores,expected_scores)
    # Independent Eq.29 oracle, explicitly enumerating each feasible next addition.
    def objective(team):
        return sum(expected_scores[i]-.2*costs[i] for i in team)+.1*sum(
            1-sim[i,j] for i in team for j in team if i<j)
    expected=[]
    for _ in range(2):
        remaining=[i for i in [0,2] if i not in expected]
        best=max(remaining,key=lambda i:(objective(expected+[i])-objective(expected),-i))
        if objective(expected+[best])-objective(expected)<0: break
        expected.append(best)
    assert chosen == expected
    assert team_objective(chosen,scores,sim,costs,.1,.2) == pytest.approx(objective(chosen))
    simple = CombinatorialLinUCB(2,1,alpha=0,diversity=0,cost=1)
    simple.b[:] = 2
    assert simple.select([[1],[1]],2,[True,True],np.eye(2),[1,3])[0] == [0]


def test_eq12_weights_and_eq30_33_discounted_credit_are_distinct_from_ucb():
    model=CombinatorialLinUCB(3,7,rho=.9,alpha=1)
    contexts=np.arange(21,dtype=float).reshape(3,7)/10
    means,scores=model.scores(contexts)
    assert scores[0] != scores[2]
    weights=sequence_weights(means,[0,2],beta=1)
    assert np.allclose(weights,[.5,.5])  # Equal utility despite different uncertainty.
    assert np.allclose(sequence_weights([1000,1001],[0,1],1),[1/(1+np.e),np.e/(1+np.e)])
    before=copy.deepcopy(model)
    model.update([0,2],contexts,2,weights)
    model.update([0],contexts,-1,[1])
    assert np.allclose(model.A[0],.81*np.eye(7)+1.9*np.outer(contexts[0],contexts[0]))
    assert np.allclose(model.b[0],-.1*contexts[0])
    assert np.array_equal(model.A[1],before.A[1])
    assert np.array_equal(model.b[1],before.b[1])


@pytest.mark.parametrize('unit',['measured_prefill_seconds','relative_excess_prefill'])
def test_dense_preparation_uses_each_sample_seven_features_and_measured_costs(monkeypatch,unit):
    from bandit_mopd import train
    student = torch.tensor([[.4,.3,.3],[.2,.5,.3]]).log()
    teacher_probs = [torch.tensor([[.6,.2,.2],[.1,.7,.2]]).log(),
                     torch.tensor([[.2,.4,.4],[.3,.3,.4]]).log()]
    ids=torch.tensor([[0,1,0,1]])
    monkeypatch.setattr(train,'rollout',lambda *a:(ids,2,'text',False))
    monkeypatch.setattr(train,'log_probs',lambda *a:student)
    def scoring(spec,samples,cfg):
        i=spec['index']
        return [teacher_probs[i] for _ in samples], [dict(prefill_seconds=[[1,3],[100,200]][j][i],
            load_and_prefill_seconds=100) for j in range(len(samples))]
    monkeypatch.setattr(train,'score_teacher_batch',scoring)
    rows=[row('math','algebra'),row('math','integral')]
    domain_model=PromptDomainModel(['math','code'],hash_bins=128).fit(rows+[row('code','python')])
    cfg=dict(context=dict(cost=unit),teachers=[dict(index=i,name=d,domain=d) for i,d in enumerate(['math','code'])])
    samples,contexts,kls,sim,costs,timing=train.prepare_batch(None,None,rows,cfg,domain_model)
    assert contexts.shape==(2,7)
    assert np.allclose(contexts,np.mean([s['contexts'] for s in samples],axis=0))
    assert np.allclose(costs,np.mean([routing_costs([1,3],cfg),routing_costs([100,200],cfg)],axis=0))
    assert np.allclose(contexts[:,6],costs)
    expected_sim=distribution_similarity(teacher_probs)
    assert np.allclose(contexts[:,5],[expected_sim[0,1]]*2)
    response=ids[0,2:,None]
    assert np.allclose(contexts[:,1],[p.gather(1,response).mean().item() for p in teacher_probs])
    assert np.allclose(contexts[:,2],[(student.exp()*(student-p)).sum(-1).mean().item() for p in teacher_probs])
    assert np.allclose(contexts[:,3],[-(p.exp()*p).sum(-1).mean().item() for p in teacher_probs])
    assert np.allclose(contexts[:,4],contexts[:,1]-student.gather(1,response).mean().item())
