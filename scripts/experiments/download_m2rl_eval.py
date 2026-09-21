"""Download version-locked M2RL evaluation sources without changing their data.

Each dataset completes independently. Restricted sources are recorded as blocked
and can be retried after the user authorizes access and logs in to Hugging Face.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile
from huggingface_hub import hf_hub_download


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def public_gpqa(target,download=False):
    provenance=target/'public_release_manifest.json'
    if provenance.exists():
        result=json.loads(provenance.read_text())
        if all((target/name).is_file() and digest(target/name)==entry['sha256']
               for name,entry in result['files'].items()):
            (target/'download_manifest.json').write_text(json.dumps(result,indent=2))
            return result
    if not download: return None
    revision='56686c06f5e19865c153de0fdb11be3890014df7'
    url=f'https://raw.githubusercontent.com/idavidrein/gpqa/{revision}/dataset.zip'
    temp=target/'dataset.zip.part'
    with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
        for chunk in iter(lambda:response.read(1024*1024),b''): f.write(chunk)
    temp.replace(target/'dataset.zip')
    with zipfile.ZipFile(target/'dataset.zip') as z:
        # The author publishes this password in the public README. Read only the
        # selected CSV rather than extracting arbitrary archive paths.
        data=z.read('dataset/gpqa_diamond.csv',pwd=b'deserted-untie-orchid')
    (target/'gpqa_diamond.csv').write_bytes(data)
    result={'name':'gpqa_diamond','status':'complete_official_public_release',
            'source':'https://github.com/idavidrein/gpqa','revision':revision,
            'files':{f:{'bytes':(target/f).stat().st_size,'sha256':digest(target/f)}
                     for f in ['dataset.zip','gpqa_diamond.csv']}}
    provenance.write_text(json.dumps(result,indent=2))
    (target/'download_manifest.json').write_text(json.dumps(result,indent=2))
    return result


def fetch(source,root):
    name=source['name']
    target=root/name
    target.mkdir(parents=True,exist_ok=True)
    if name=='gpqa_diamond':
        existing=public_gpqa(target)
        if existing: return existing
    manifest={'name':name,'repo':source['repo'],'revision':source['revision'],'files':{},'status':'downloading'}
    for file in source['files']:
        try:
            p=Path(hf_hub_download(source['repo'],file,repo_type='dataset',revision=source['revision'],local_dir=target))
            manifest['files'][file]={'bytes':p.stat().st_size,'sha256':digest(p)}
            print(json.dumps({'dataset':name,'file':file,'status':'verified','bytes':p.stat().st_size}),flush=True)
        except Exception as error:
            if name=='gpqa_diamond' and 'Gated' in type(error).__name__:
                return public_gpqa(target,download=True)
            # Do not serialize request headers, tokens, or signed download URLs.
            manifest['status']='blocked_access' if 'Gated' in type(error).__name__ else 'download_failed'
            manifest['failed_file']=file
            manifest['error_type']=type(error).__name__
            break
        (target/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    else:
        manifest['status']='complete'
    (target/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiments/eval_sources.lock.json')
    p.add_argument('--output',default='data/eval/raw')
    p.add_argument('--only',nargs='*')
    args=p.parse_args()
    sources=json.loads(Path(args.config).read_text())['sources']
    if args.only: sources=[s for s in sources if s['name'] in args.only]
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,s,root) for s in sources]
        for f in as_completed(futures):
            result=f.result(); results.append(result)
            print(json.dumps({'dataset':result['name'],'status':result['status']}),flush=True)
    print(json.dumps({'completed':[r['name'] for r in results if r['status'].startswith('complete')],
                      'pending':[r['name'] for r in results if not r['status'].startswith('complete')]},indent=2),flush=True)


if __name__=='__main__': main()
