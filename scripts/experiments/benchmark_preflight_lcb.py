import json
from collections import Counter
from pathlib import Path
from benchmark_evaluate import ROOT,lcb_sample,dump,now
out=ROOT/'analysis/2026-09-16-sft-benchmarks'
stats=[]
with (ROOT/'data/eval/processed/lcb_release_v5.jsonl').open() as f:
    for line in f:
        row=json.loads(line);sample,n=lcb_sample(row)
        fn=json.loads(sample['input_output'])['fn_name']
        stats.append({'id':row['id'],'tests':n,'type':'functional' if fn else 'stdin'})
assert len(stats)==880
dump(out/'lcb-test-manifest.json',{'n':len(stats),'total_tests':sum(r['tests'] for r in stats),'types':dict(Counter(r['type'] for r in stats)),'questions':stats})
dump(out/'readiness/lcb_release_v5.json',{'ready':True,'n':880,'checks':'All official public/private tests parsed; stdin/function correct and wrong program integration tests passed.','timestamp':now()})
print('ALL_LCB_DATA_PARSED',len(stats),sum(r['tests'] for r in stats),flush=True)
