"""Sequential, matched student-update-budget comparison; fresh outputs only."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/experiments/bandit_full_pilot.json')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--steps', type=int, default=100)
    a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(a.config).read_text())
    for seed in map(int,a.seeds.split(',')):
        for mode in ('fixed','random','uniform','bandit'):
            cfg=copy.deepcopy(base)
            cfg.update(seed=seed,mode=mode,steps=a.steps,output=str(root/f'{mode}-seed{seed}'),
                       save_optimizer=False)
            config_path=root/f'{mode}-seed{seed}.json'
            config_path.write_text(json.dumps(cfg,indent=2))
            with (root/f'{mode}-seed{seed}.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-m','bandit_mopd.train','--config',str(config_path)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    main()
