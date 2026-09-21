"""Delete explicitly labelled Agent/workbench rows and dedicated BFCL datasets.

Preserve every retained JSONL byte and split assignment; record before/after
hashes rather than retaining the deleted data. Use --apply for the mutation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def is_agent(row):
    return (row.get('domain') == 'agent'
            or row.get('dataset') == 'nano_v3_sft_profiled_workbench')


def filter_agent_jsonl(path, apply=False):
    path = Path(path)
    if path.is_symlink(): raise ValueError(f'Refuse dataset symlink: {path}')
    before, after = hashlib.sha256(), hashlib.sha256()
    total = removed = removed_bytes = 0
    with path.open('rb') as f:
        for line in f:
            before.update(line)
            if not line.strip():
                after.update(line); continue
            total += 1
            if is_agent(json.loads(line)):
                removed += 1; removed_bytes += len(line)
            else: after.update(line)
    entry = dict(path=str(path), total_rows=total, removed_rows=removed,
                 retained_rows=total-removed, removed_bytes=removed_bytes,
                 before_sha256=before.hexdigest(), after_sha256=after.hexdigest())
    if apply and removed:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.without-agent-',delete=False) as out:
            temp = Path(out.name)
            check = hashlib.sha256()
            try:
                with path.open('rb') as src:
                    for line in src:
                        check.update(line)
                        if not line.strip() or not is_agent(json.loads(line)): out.write(line)
                out.flush()
                if check.hexdigest() != entry['before_sha256']:
                    raise RuntimeError(f'Dataset changed during filtering: {path}')
                temp.chmod(path.stat().st_mode & 0o777)
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    return entry


def purge(root, apply=False):
    root = Path(root).resolve()
    report = {'applied': apply, 'files': [], 'deleted_datasets': []}
    dedicated = [root/'eval/raw/bfcl_v3', root/'eval/processed/bfcl_v3_categories.json']
    for path in dedicated:
        if not path.exists(): continue
        if path.is_symlink(): raise ValueError(f'Refuse dataset symlink: {path}')
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size
        report['deleted_datasets'].append({'path':str(path),'bytes':size})
        if apply:
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
    paths = sorted({p for folder in ('raw','processed','eval/processed')
                    for p in (root/folder).rglob('*.jsonl')})
    for path in paths:
        entry = filter_agent_jsonl(path, apply)
        report['files'].append(entry)
        print(json.dumps(entry),flush=True)
    report['removed_rows'] = sum(f['removed_rows'] for f in report['files'])
    report['removed_bytes'] = sum(f['removed_bytes'] for f in report['files']) + sum(f['bytes'] for f in report['deleted_datasets'])
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', required=True)
    args=parser.parse_args()
    result=purge(args.data_root,args.apply)
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','deleted_datasets')}),flush=True)
