import numpy as np
import pytest
from bandit_mopd.context import routing_costs
from bandit_mopd.core import CombinatorialLinUCB

CFG={'context':{'cost':'relative_excess_prefill'}}


def test_relative_cost_is_bounded_and_ignores_shared_slowdown():
    assert np.allclose(routing_costs([10,20,40],CFG),[0,.25,.75])
    for seconds in ([0,0,0],[1e6]*3,[123]):
        assert np.array_equal(routing_costs(seconds,CFG),np.zeros(len(seconds)))
    raw=np.array([1.,2.,3.,1000.])
    for multiplier in (.001,1,100,1e6):
        cost=routing_costs(raw*multiplier,CFG)
        assert np.allclose(cost,routing_costs(raw,CFG))
        assert ((cost>=0)&(cost<=1)).all()
    assert np.all(routing_costs(raw+1000,CFG)<=routing_costs(raw,CFG))


@pytest.mark.parametrize('values',[[],[-1,1],[float('inf'),1],[float('nan'),1],[[1,2]]])
def test_invalid_raw_cost_rejected(values):
    with pytest.raises(ValueError):routing_costs(values,CFG)


def test_runtime_units_do_not_change_ucb_or_teacher_selection():
    contexts=np.array([[1,-.4,.01,.5,.01,.99,0],[0,-.5,.02,.6,.02,.98,0]],dtype=float)
    results=[]
    for times in ([100,120],[10000,12000]):
        costs=routing_costs(times,CFG);contexts[:,6]=costs
        b=CombinatorialLinUCB(2,7,alpha=.5,cost=.001)
        b.update([0,1],contexts,-.03,[.5,.5])
        results.append(b.select(contexts,2,[True,True],np.eye(2),costs))
    assert results[0][0]==results[1][0]
    assert np.allclose(results[0][2],results[1][2])
