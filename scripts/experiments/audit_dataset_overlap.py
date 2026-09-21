"""Audit full-question containment between training prompts and frozen eval.

Case/Unicode/whitespace normalization only: mathematical operators stay intact.
This is an exact-text screen, not proof against semantic/near-duplicate leakage.
Raw training data is preserved; an optional clean view excludes exact matches.
"""
import argparse
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()


class QuestionMatcher:
    def __init__(self,examples):
        self.nodes=[{}]; self.fail=[0]; self.outputs=[[]]; self.questions=[]
        seen=set(); self.short=[]
        for row in examples:
            key=(row['benchmark'],row['id'])
            if key in seen: continue
            seen.add(key)
            text=normalized(row['question'])
            if not text: continue
            index=len(self.questions)
            self.questions.append((text,key))
            if len(text)<40:
                self.short.append(index); continue
            state=0
            for char in text[:64]:
                if char not in self.nodes[state]:
                    self.nodes[state][char]=len(self.nodes)
                    self.nodes.append({}); self.fail.append(0); self.outputs.append([])
                state=self.nodes[state][char]
            self.outputs[state].append(index)
        queue=deque(self.nodes[0].values())
        while queue:
            parent=queue.popleft()
            for char,child in self.nodes[parent].items():
                queue.append(child); suffix=self.fail[parent]
                while suffix and char not in self.nodes[suffix]: suffix=self.fail[suffix]
                self.fail[child]=self.nodes[suffix].get(char,0)
                self.outputs[child].extend(self.outputs[self.fail[child]])

    def match(self,messages):
        text=normalized('\n'.join(m['content'] for m in messages))
        candidates=set(self.short); state=0
        for char in text:
            while state and char not in self.nodes[state]: state=self.fail[state]
            state=self.nodes[state].get(char,0)
            candidates.update(self.outputs[state])
        return [self.questions[i][1] for i in candidates
                if (self.questions[i][0]==text if i in self.short else self.questions[i][0] in text)]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-dir',default='data/processed/nemotron')
    p.add_argument('--eval-dir',default='data/eval/processed')
    p.add_argument('--report',default='results/data-prepare/overlap-audit.json')
    p.add_argument('--clean-dir')
    a=p.parse_args(); eval_dir=Path(a.eval_dir)
    evaluation_manifest=json.loads((eval_dir/'manifest.json').read_text())
    if evaluation_manifest['pending']:
        raise ValueError('Finish all evaluation downloads before the final overlap audit')
    names=['aime24','aime25','gpqa_diamond','hle_text','ifeval','ifbench','lcb_release_v6']
    examples=[]
    for name in names:
        with (eval_dir/f'{name}.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                examples.append({k:row[k] for k in ('benchmark','id','question')})
    matcher=QuestionMatcher(examples)
    train_dir=Path(a.train_dir)
    clean=Path(a.clean_dir) if a.clean_dir else None
    if clean: clean.mkdir(parents=True,exist_ok=False)
    report={'method':'NFKC/case/whitespace normalized complete-question containment; short questions require exact prompt equality',
            'limitation':'Does not detect semantic paraphrases, changed options or every formatting variant.',
            'eval_manifest_sha256':digest(eval_dir/'manifest.json'),'splits':{},'overlaps':[]}
    all_ids=set()
    for split in ('train','probe','dev'):
        count=0; hits=0; counts=Counter(); path=train_dir/f'{split}.jsonl'
        dest=(clean/f'{split}.jsonl').open('w') if clean else None
        with path.open() as f:
            for line in f:
                row=json.loads(line); count+=1
                if count%10000==0: print(f'{split}: scanned {count} records',flush=True)
                if row['id'] in all_ids: raise ValueError('Duplicate train/probe/dev fingerprint')
                all_ids.add(row['id'])
                overlaps=matcher.match(row['messages'])
                if overlaps:
                    hits+=1
                    report['overlaps'].append({'split':split,'id':row['id'],'domain':row['domain'],
                                               'matches':overlaps})
                else:
                    counts[row['domain']]+=1
                    if dest: dest.write(line)
        if dest: dest.close()
        report['splits'][split]={'input':count,'overlapping_prompts':hits,'retained':count-hits,
                                'retained_by_domain':dict(counts),'input_sha256':digest(path)}
        if clean: report['splits'][split]['output_sha256']=digest(clean/f'{split}.jsonl')
        print(split,report['splits'][split],flush=True)
    if clean:
        (clean/'manifest.json').write_text(json.dumps({'parent_manifest_sha256':digest(train_dir/'manifest.json'),
            'counts':{s:v['retained_by_domain'] for s,v in report['splits'].items()},
            'split_sha256':{s:v['output_sha256'] for s,v in report['splits'].items()},
            'overlap_report':str(Path(a.report)),'teacher_exposure':'May have been used during teacher training'},indent=2))
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
