"""Relocatable source manifests for the separate routed-teacher campaign."""
import json
from pathlib import Path


def resolve_sources(manifest, history, seed):
    manifest = Path(manifest).expanduser().resolve()
    records = json.loads(manifest.read_text())[f't{history}-s{seed}']
    for record in records.values():
        for key in ('checkpoint', 'summary'):
            if record.get(key):
                path = Path(record[key]).expanduser()
                record[key] = str(path if path.is_absolute() else
                                  (manifest.parent / path).resolve())
    # metadata.initial_checkpoint is a provenance identity embedded in the
    # checkpoint, not a file to open. Preserve it when moving the artifacts.
    return records
