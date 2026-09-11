#!/usr/bin/env python3
"""Fetch or verify only the weights named in the project's pinned manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Weight size mismatch: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != entry['sha256']:
        raise ValueError(f'Weight checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'verify'))
    parser.add_argument('repo', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('fold', choices=['all'] + [str(i) for i in range(12)])
    args = parser.parse_args()
    manifest = json.loads((args.repo / 'reproductions/wsts_res18_unet_t1/official_weights_manifest.json').read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['weights']:
        if args.fold != 'all' and str(entry['fold_id']) != args.fold:
            continue
        path = args.destination / entry['filename']
        if args.action == 'fetch' and not path.exists():
            url = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{entry['hub_path']}"
            partial = path.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=180) as response, partial.open('wb') as handle:
                shutil.copyfileobj(response, handle)
            verify(partial, entry)
            partial.replace(path)
        verify(path, entry)
        print(f"VERIFIED fold={entry['fold_id']} sha256={entry['sha256']} {path}")

if __name__ == '__main__':
    main()
