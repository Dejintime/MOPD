"""Produce explicit evaluation views; raw benchmark records remain immutable.

Evaluation files are never used as bandit feedback or student training data.
The source lists all nine M2RL benchmark names, but does not publish one complete
evaluation split manifest. Ambiguous LCB views are therefore named separately.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import pyarrow.parquet as pq


def read_jsonl(path):
    with open(path,encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def record(benchmark,index,domain,question,prompt,answer,metadata=None):
    return {'id':f'{benchmark}:{index}','benchmark':benchmark,'domain':domain,
            'question':question,'messages':[{'role':'user','content':prompt}],
            'answer':answer,'metadata':metadata or {},'usage':'final_evaluation_only'}


def math_record(name,row,i):
    question=row['problem']
    if 'answer' in row:
        answer=str(row['answer']).strip()
    else:
        match=re.fullmatch(r'\\boxed\{([^{}]+)\}',row['solution'].strip())
        if not match: raise ValueError('Unrecognized AIME solution format')
        answer=match.group(1)
    if not re.fullmatch(r'\d{1,3}',answer): raise ValueError('AIME answer must be an integer 0-999')
    prefix=('Solve the following math problem step by step. The last line of your response should '
            'be of the form Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer '
            'to the problem.\n\n')
    return record(name,row.get('id',i),'math',question,prefix+question,answer,{'rm_type':'deepscaler'})


def gpqa_record(row,i,seed=42):
    # Match M2RL's insertion procedure, replacing its unseeded randint with a
    # stable per-question seed. The correct-option mapping is stored explicitly.
    question=row['Question'].strip()
    rng=random.Random(hashlib.sha256(f'{seed}:{question}'.encode()).hexdigest())
    gold=rng.randrange(4)
    choices=[row[f'Incorrect Answer {j}'].strip() for j in (1,2,3)]
    choices.insert(gold,row['Correct Answer'].strip())
    prompt=('Answer the following multiple choice question. The last line of your response '
            "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER "
            'is one of ABCD. Think step by step before answering.\n\n'+question+'\n\n'+
            '\n'.join(f'{letter}) {choice}' for letter,choice in zip('ABCD',choices)))
    return record('gpqa_diamond',i,'science',question,prompt,'ABCD'[gold],
                  {'rm_type':'gpqa','choices':choices,'correct_letter':'ABCD'[gold],
                   'shuffle_seed':seed,'source_row':i})


def lcb_record(row,version):
    question=row['question_content']
    prompt='Please solve the following coding problem in Python. Return the final solution in a ```python code block.\n\n'+question
    if row.get('starter_code'): prompt+='\n\nStarter code:\n'+row['starter_code']
    return record('livecodebench',row['question_id'],'code',question,prompt,None,
                  {**row,'incremental_version':version,'rm_type':'livecodebench',
                   'prompt_status':'adapter template; use official LCB runner for published scoring'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='data/eval/raw')
    p.add_argument('--output',default='data/eval/processed')
    a=p.parse_args(); raw=Path(a.raw); out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    summary={'datasets':{},'pending':[], 'm2rl_evaluation_manifest':'not fully published',
             'evaluation_only':True}
    def save(name,rows,notes):
        path=out/f'{name}.jsonl'
        ids=[r['id'] for r in rows]
        if len(set(ids))!=len(ids): raise ValueError(f'Duplicate evaluation IDs in {name}')
        temp=path.with_suffix('.jsonl.part')
        with temp.open('w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        temp.replace(path)
        summary['datasets'][name]={'rows':len(rows),'file':str(path),'sha256':sha(path),'notes':notes}
        print(name,len(rows),flush=True)
    for name,file in [('aime24','test-00000-of-00001.parquet'),('aime25','test.jsonl')]:
        path=raw/name/file
        if not path.exists(): summary['pending'].append(name); continue
        rows=pq.read_table(path).to_pylist() if path.suffix=='.parquet' else read_jsonl(path)
        if len(rows)!=30: raise ValueError(f'{name}: expected 30 questions')
        save(name,[math_record(name,r,i) for i,r in enumerate(rows)],'30 official-year questions; M2RL math prompt')
    path=raw/'gpqa_diamond/gpqa_diamond.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        if len(rows)!=198: raise ValueError('GPQA-Diamond expected 198 rows')
        hf_manifest=raw/'gpqa_diamond/download_manifest.json'
        if hf_manifest.exists() and json.loads(hf_manifest.read_text())['status']=='complete':
            provenance='Version-locked official Hugging Face source'
        else:
            release=json.loads((raw/'gpqa_diamond/public_release_manifest.json').read_text())
            if release['files']['gpqa_diamond.csv']['sha256']!=sha(path):
                raise ValueError('GPQA public-release CSV hash mismatch')
            provenance='Official author public archive at '+release['revision']
        save('gpqa_diamond',[gpqa_record(r,i) for i,r in enumerate(rows)],
             provenance+'; 198 rows; deterministic choice order; differs from unseeded M2RL choice shuffling')
    else: summary['pending'].append('gpqa_diamond')
    for name,file in [('ifeval','ifeval_input_data.jsonl'),('ifbench','data/train-00000-of-00001.parquet')]:
        path=raw/name/file
        if not path.exists(): summary['pending'].append(name); continue
        rows=pq.read_table(path).to_pylist() if path.suffix=='.parquet' else read_jsonl(path)
        normalized=[]
        for i,r in enumerate(rows):
            if len(r['instruction_id_list'])!=len(r['kwargs']): raise ValueError('Instruction metadata length mismatch')
            normalized.append(record(name,r.get('key',i),'if',r['prompt'],r['prompt'],None,
                {'rm_type':name,'instruction_id_list':r['instruction_id_list'],'kwargs':r['kwargs'],'prompt_text':r['prompt']}))
        save(name,normalized,'Full official source; preserve all instructions/kwargs without dropping unsupported cases')
    path=raw/'hle/data/test-00000-of-00001.parquet'
    if path.exists():
        rows=pq.read_table(path).to_pylist()
        normalized=[]; image_count=0
        for i,r in enumerate(rows):
            if r.get('image'): image_count+=1; continue
            normalized.append(record('hle_text',r.get('id',i),'science',r['question'],r['question'],r['answer'],
                {**{k:v for k,v in r.items() if k not in ['question','answer','image','image_preview','rationale_image']},
                 'rationale_image_present':bool(r.get('rationale_image'))}))
        save('hle_text',normalized,f'Text-only Qwen3 view; {image_count} image-bearing rows excluded; full raw parquet retained')
    else: summary['pending'].append('hle_access_required')
    lcb_paths=[raw/'livecodebench'/('test.jsonl' if i==1 else f'test{i}.jsonl') for i in range(1,7)]
    if all(path.exists() for path in lcb_paths):
        versions=[[lcb_record(r,i+1) for r in read_jsonl(path)] for i,path in enumerate(lcb_paths)]
        all_rows=sum(versions,[])
        for name,rows,notes in [
            ('lcb_release_v5',sum(versions[:5],[]),'Official cumulative release_v5'),
            ('lcb_release_v6',all_rows,'Official cumulative release_v6'),
            ('lcb_v5_increment',versions[4],'Only test5.jsonl; not cumulative release_v5'),
            ('lcb_v6_increment',versions[5],'Only test6.jsonl; not cumulative release_v6'),
            ('lcb_v5_m2rl_window',[r for r in all_rows if '2024-07-01'<=r['metadata']['contest_date'][:10]<'2025-02-01'],
             'Window from M2RL bundled Gym validation config: 2024-07-01 <= date < 2025-02-01; full paper test-ID manifest unavailable')]:
            save(name,rows,notes)
    else: summary['pending'].append('livecodebench_download')
    # Agent/BFCL is outside the user-requested four-domain scope.

    (out/'manifest.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
