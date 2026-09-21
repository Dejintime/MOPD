"""Check every frozen IF constraint before declaring the queue stage ready."""
import json, os, signal, sys
from pathlib import Path
from benchmark_evaluate import load_if,if_score,ROOT,read_rows,dump,now,alarm_handler
os.environ['NLTK_DATA']='/home/yangdejin/.mopd-benchmark-nltk'
signal.signal(signal.SIGALRM,alarm_handler)
for name in ('ifeval','ifbench'):
    lib=load_if(name);rows=read_rows(ROOT/f'data/eval/processed/{name}.jsonl')
    constraints=0
    for row in rows:
        signal.alarm(30)
        try:
            if_score(lib,row,'A simple apple example. Second sentence.\n\nAnother paragraph.')
            constraints+=len(row['metadata']['instruction_id_list'])
        finally:signal.alarm(0)
    result={'ready':True,'n':len(rows),'constraints':constraints,'checks':'All source instructions instantiated and strict/loose checks executed on synthetic response; no benchmark predictions used.','timestamp':now()}
    dest=ROOT/'analysis/2026-09-16-sft-benchmarks/readiness'/f'{name}.json';dump(dest,result);print(json.dumps(result),flush=True)
