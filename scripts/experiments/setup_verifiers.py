"""Install the pinned WASI Python artifact without Docker or administrator access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

URL = 'https://github.com/brettcannon/cpython-wasi-build/releases/download/v3.13.15/python-3.13.15-wasi_sdk-24.zip'
SHA256 = '67e1c32a85d5e0600c0939f5fd4023ca0127f1a9e81f5499697467bda67af78d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/verifier_runtime/python-3.13.15-wasi'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Runtime already exists: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.wasi-setup-') as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp)/'python.zip'
            with urllib.request.urlopen(URL, timeout=60) as source, archive.open('wb') as target:
                shutil.copyfileobj(source, target)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise ValueError('WASI archive SHA256 differs from published release digest')
        root = Path(tmp)/'runtime'
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f'Unsafe archive entry: {p}')
            z.extractall(root)
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file()}
        (root/'manifest.json').write_text(json.dumps({'url': URL, 'archive_sha256': SHA256,
            'python_version': '3.13.15', 'wasi_sdk': 24, 'files_sha256': files}, indent=2))
        root.rename(args.output)
    print(f'Installed verified WASI runtime: {args.output}')


if __name__ == '__main__':
    main()
