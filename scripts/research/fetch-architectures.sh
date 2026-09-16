#!/usr/bin/env bash
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Architecture asset downloads require a Slurm allocation.' >&2
  exit 2
fi
: "${WILDFIRE_ROOT:?set WILDFIRE_ROOT to the configured external artifact root}"
python - <<'PY'
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

root = Path(os.environ['WILDFIRE_ROOT']).expanduser().resolve()
revision = '3bb39e8739149c3777d0325349b2a6c32c6413db'
assets = (
    ('cache/swin/swin_tiny_patch4_window7_224.pth',
     'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
     '9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3'),
    ('cache/huggingface/nvidia-mit-b2-3bb39e87/config.json',
     f'https://huggingface.co/nvidia/mit-b2/resolve/{revision}/config.json',
     'd9a879499e7d73e2b33af0638cee320b1070c8f0dadb620eac8907df3d18caa9'),
    ('cache/huggingface/nvidia-mit-b2-3bb39e87/pytorch_model.bin',
     f'https://huggingface.co/nvidia/mit-b2/resolve/{revision}/pytorch_model.bin',
     '4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f'),
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


for relative, url, expected in assets:
    destination = root / relative
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or digest(destination) != expected:
            raise ValueError(f'existing asset differs; refusing overwrite: {destination}')
        print(f'VERIFIED {expected} {destination}', flush=True)
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.',
                                             suffix='.partial', dir=destination.parent)
    partial = Path(temporary)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'wildfire-research-reproduction/1.0'})
        with os.fdopen(descriptor, 'wb') as output:
            with urllib.request.urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest(partial) != expected:
            raise ValueError(f'downloaded asset SHA256 mismatch: {url}')
        # Same-filesystem hard-link publication is atomic and fails if another
        # writer created the destination. Never replace even a bad existing file.
        os.link(partial, destination)
        print(f'DOWNLOADED {expected} {destination}', flush=True)
    finally:
        partial.unlink(missing_ok=True)
PY
