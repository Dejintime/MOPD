"""Cost-scale regression on archived observations; NOT on-policy training.

Query each policy but update the historically selected team with its observed
non-cost reward and credit weights. Counterfactual team rewards are unavailable.
Repeating observations is a synthetic stability test, not new training evidence.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from bandit_mopd.context import routing_costs
from bandit_mopd.core import CombinatorialLinUCB


def observation(event,cfg,multiplier=1):
    samples=np.asarray(event['sample_contexts'],dtype=float).copy()
    for contexts in samples:
        contexts[:,6]=routing_costs(contexts[:,6]*multiplier,cfg)
    contexts=samples.mean(0);costs=contexts[:,6].copy()
    names=[t['name'] for t in cfg['teachers']]
    chosen=[names.index(name) for name in event['selected']]
    terms=event['reward_terms'];rc=cfg['reward']
    reward=(rc.get('task',0)*terms['task']+rc['distill']*terms['distill']
            +rc['stability']*terms['stability']-rc['redundancy']*terms['redundancy']
            -rc['cost']*costs[chosen].sum())
    return contexts,costs,chosen,reward


def replay(events,cfg,count,multiplier=1,verify_original=False):
    bandit=CombinatorialLinUCB(len(cfg['teachers']),7,cfg['alpha'],cfg['rho'],
                              diversity=cfg['diversity'],cost=cfg['selection_cost'])
    decisions=[];first=[]
    for j in range(count):
        event=events[j%len(events)];contexts,costs,historical_team,reward=observation(event,cfg,multiplier)
        chosen,means,scores=bandit.select(contexts,cfg['k'],np.asarray(event['candidate_kl_max'])<=cfg['kl_gate'],np.asarray(event['teacher_similarity']),costs)
        if verify_original and j<len(events):
            assert np.allclose(scores,event['ucb'],atol=1e-8,rtol=1e-8)
            assert chosen==historical_team
            assert abs(reward-event['reward'])<1e-8
        decisions.append(chosen)
        if j<len(events):first.append(dict(observed_step=event['optimizer_step'],queried_team=chosen,
            historical_team=historical_team,observed_team_adjusted_reward=float(reward),
            initial_gains=(scores-cfg['selection_cost']*costs).tolist(),
            context_cost=costs.tolist(),reward_cost_penalty=float(-cfg['reward']['cost']*costs[historical_team].sum())))
        bandit.update(historical_team,contexts,float(reward),event['weights'])
    means,scores=bandit.scores(contexts)
    return dict(queried_batches=count,empty_queries=sum(not c for c in decisions),
        decisions=decisions,first_observations=first,
        last_context_requery_initial_gains=(scores-cfg['selection_cost']*costs).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--config',required=True)
    p.add_argument('--baseline-config',required=True);p.add_argument('--output',required=True);p.add_argument('--stress-updates',type=int,default=200)
    args=p.parse_args()
    events=[e for e in map(json.loads,Path(args.metrics).read_text().splitlines()) if e['status']=='updated']
    if not events or args.stress_updates < 1:raise ValueError('Need successful observations and positive stress-updates')
    if any(e.get('context_features',[])[-1:]!=['teacher_prefill_seconds'] for e in events):
        raise ValueError('This diagnostic expects historical raw-second cost contexts')
    cfg=json.loads(Path(args.config).read_text());old=json.loads(Path(args.baseline_config).read_text())
    result=dict(method=__doc__,observations=len(events),missing_failed_step_contexts=True,
        historical=replay(events,old,len(events),verify_original=True),
        corrected=replay(events,cfg,len(events)),
        historical_stress=replay(events,old,args.stress_updates),
        corrected_stress=replay(events,cfg,args.stress_updates),
        corrected_100x_slower=replay(events,cfg,args.stress_updates,multiplier=100))
    assert result['corrected_stress']['decisions']==result['corrected_100x_slower']['decisions']
    assert np.allclose(result['corrected_stress']['last_context_requery_initial_gains'],result['corrected_100x_slower']['last_context_requery_initial_gains'],atol=1e-8)
    assert result['corrected_stress']['empty_queries']==0
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({key:{k:v for k,v in value.items() if k in ('empty_queries','last_context_requery_initial_gains')} for key,value in result.items() if isinstance(value,dict)},indent=2))


if __name__=='__main__':main()
