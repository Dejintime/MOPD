"""Download pinned blend, restore math references, normalize and split.

Run from the project root with the server's existing mopd Python environment.
Original-source revisions are resolved ONCE and written to sources.lock.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
from bandit_mopd.data import prepare
from bandit_mopd.preflight import sha256
from scripts.experiments.remove_agent_data import filter_agent_jsonl

BLEND = 'nvidia/Nemotron-3-Nano-RL-Training-Blend'
REVISION = 'ffd169f2b74bb492ec607d64bd56f7435054972b'
PUBLISHED_SHA256 = '3c027483436670a3814a39433dc65938a3ef06923e680f26ee293831e5100cb7'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/processed/nemotron')
    parser.add_argument('--raw-dir', default='data/raw/nemotron')
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    lock_path = raw/'sources.lock.json'
    helper_path = Path('third_party/nemotron/create_nanov3_jsonl.py')
    spec = importlib.util.spec_from_file_location('nvidia_restore', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock[BLEND] != REVISION:
            raise ValueError('Blend revision mismatch')
    else:
        api = HfApi()
        lock = {BLEND: REVISION}
        for cfg in helper.TARGET_DATASETS.values():
            name = cfg['hf_dataset']
            lock[name] = api.dataset_info(name).sha
        lock_path.write_text(json.dumps(lock, indent=2))
    source = hf_hub_download(BLEND, 'train.jsonl', repo_type='dataset', revision=REVISION,
                             local_dir=raw)
    source_sha = sha256(source)
    filtered_provenance = raw/'agent_filter.json'
    filtered_verified = False
    if filtered_provenance.exists():
        audit = json.loads(filtered_provenance.read_text())
        filtered_verified = any(entry['before_sha256'] == PUBLISHED_SHA256
            and entry['after_sha256'] == source_sha
            and Path(entry['path']).resolve() == Path(source).resolve()
            for entry in audit['files'])
    if source_sha != PUBLISHED_SHA256 and not filtered_verified:
        raise ValueError('Raw source is neither the pinned upstream object nor its audited Agent-filtered derivative')
    if source_sha == PUBLISHED_SHA256:
        entry = filter_agent_jsonl(source, apply=True)
        filtered_provenance.write_text(json.dumps({'applied':True,'files':[entry]},indent=2))
        source_sha = entry['after_sha256']
        filtered_verified = True
    datasets = {}
    for name, cfg in helper.TARGET_DATASETS.items():
        print('Restoring source:', cfg['hf_dataset'], flush=True)
        if cfg['hf_dataset']=='Skywork/Skywork-OR1-RL-Data':
            # load_dataset(split='math') can also download/build its unused code
            # split. Fetch the exact official math shard and preserve row order.
            math_path=hf_hub_download(cfg['hf_dataset'],'data/math-00000-of-00001.parquet',
                                     repo_type='dataset',revision=lock[cfg['hf_dataset']])
            datasets[name]=Dataset.from_parquet(math_path)
        else:
            datasets[name] = load_dataset(cfg['hf_dataset'], split=cfg['split'],
                                          revision=lock[cfg['hf_dataset']])
    target = raw/'train_complete.jsonl'
    if not target.exists():
        temp = raw/'train_complete.jsonl.part'
        with open(source) as f, temp.open('w') as out:
            for line in f:
                row = json.loads(line)
                if row.get('dataset') == 'nano_v3_sft_profiled_workbench':
                    continue
                placeholder = row.get('_hf_placeholder')
                if placeholder:
                    name = row['dataset']
                    cfg = helper.TARGET_DATASETS[name]
                    row = helper.restore_record(row, datasets[name][int(placeholder['row'])],
                                                cfg['question_path'], cfg['answer_path'])
                out.write(json.dumps(row, ensure_ascii=False)+'\n')
        temp.replace(target)
    print('Normalizing and splitting complete blend',flush=True)
    report = prepare(target, args.output)
    report['source_revisions'] = lock
    report['upstream_restore_helper_sha256'] = sha256(helper_path)
    report['raw_blend_sha256'] = sha256(source)
    report['matches_published_sha256'] = source_sha == PUBLISHED_SHA256
    report['upstream_published_sha256'] = PUBLISHED_SHA256
    report['audited_agent_filtered_source'] = filtered_verified
    report['excluded_domains'] = ['agent']
    (Path(args.output)/'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
