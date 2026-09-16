"""Standard-library artifact paths shared by portable research entrypoints.

Environment paths are expanded relative to the launch working directory. A
portable entrypoint must call ``load_paths(require_explicit=True, required=...)``
before importing a tensor driver. Historical drivers retain their old defaults.
Teacher manifest checkpoint/summary paths are relative to the manifest itself.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

LEGACY_ROOT = Path('/project/6085198/kulbear/wildfire')


@dataclass(frozen=True)
class ResearchPaths:
    root: Path
    upstream: Path
    data: Path
    stats: Path
    b3_checkpoint: Path | None
    b5_checkpoint: Path | None
    teacher_manifest: Path

    def initial_checkpoint(self, history):
        if history not in (1, 5):
            raise ValueError(f'unsupported history T={history}')
        selected = self.b3_checkpoint if history == 1 else self.b5_checkpoint
        if selected is not None:
            return selected
        run = (self.root / 'runs/corrected-B3-S0-3K-21093266' if history == 1 else
               self.root / 'archive/pre-t1-cleanup-2026-09-04/runs/corrected-B5-S0-3K-21144563')
        record = json.loads((run / 'completed.json').read_text())
        checkpoint = Path(record['checkpoint'])
        if not checkpoint.is_absolute():
            return run / checkpoint
        if history == 5:
            checkpoint = run / checkpoint.relative_to(self.root / 'runs' / run.name)
        return checkpoint


def load_paths(*, require_explicit=False, required=(), environ=None):
    """Read config; required contains field names such as ``b3_checkpoint``.

    Explicit mode forbids historical defaults for the four base variables and
    selected required artifacts; existence is checked by their actual loaders.
    """
    env = os.environ if environ is None else environ
    fields = tuple(ResearchPaths.__dataclass_fields__)
    unknown = set(required) - set(fields)
    if unknown:
        raise ValueError(f'unknown path fields: {sorted(unknown)}')
    if require_explicit:
        needed = ('root', 'upstream', 'data', 'stats', *required)
        missing = sorted({'WILDFIRE_' + key.upper() for key in needed
                          if not env.get('WILDFIRE_' + key.upper(), '').strip()})
        if missing:
            raise ValueError('portable runtime requires explicit configuration: ' + ', '.join(missing))

    def selected(key, fallback=None):
        value = env.get('WILDFIRE_' + key.upper(), '').strip()
        return Path(value).expanduser().resolve() if value else fallback

    root = selected('root', LEGACY_ROOT)
    return ResearchPaths(root=root,
        upstream=selected('upstream', root / 'cache/WildfireSpreadTS-res18-runtime'),
        data=selected('data', root / 'hdf5/wstsplus-active-fixed'),
        stats=selected('stats', root / 'runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz'),
        b3_checkpoint=selected('b3_checkpoint'), b5_checkpoint=selected('b5_checkpoint'),
        teacher_manifest=selected('teacher_manifest', Path(__file__).resolve().parents[1] /
            'docs/experiments/three_directions_manifest.json'))


def resolve_teacher_records(manifest, history, seed):
    manifest = Path(manifest).expanduser().resolve()
    records = json.loads(manifest.read_text())['sources'][str(history)][str(seed)]
    for record in records.values():
        for key in ('checkpoint', 'summary'):
            if key in record:
                path = Path(record[key]).expanduser()
                record[key] = str(path if path.is_absolute() else (manifest.parent / path).resolve())
    return records


def validate_teacher_artifact(record):
    """Validate both recorded byte count and SHA-256 before deserialization."""
    path = Path(record['checkpoint'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != record['bytes'] or digest.hexdigest() != record['sha256']:
        raise ValueError(f'teacher checkpoint changed: {path}')
    return path
