"""Train the single P02 FireDrop-plus-BlockDrop rapid prototype."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from .contract import experiment_spec, upstream_arguments, validate_inventory
from .entrypoint import _install_runtime_contract, load_training_stats
from .prototype import (
    BLOCK_DROPOUT_PROBABILITY,
    BLOCK_PROTOTYPE_ID,
    FIRE_DROPOUT_PROBABILITY,
    install_training_fire_and_block_dropout,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    args = parser.parse_args(argv)

    upstream_root = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    train_path = upstream_root / "src" / "train.py"
    if not train_path.is_file():
        raise ValueError(f"pinned upstream train.py is missing: {train_path}")
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    spec = experiment_spec("C00")
    _install_runtime_contract(upstream_root, spec.experiment_id, stats)
    install_training_fire_and_block_dropout(upstream_root)

    work_root = run_root / "work"
    work_root.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(work_root)
    sys.argv = [
        str(train_path),
        *upstream_arguments(
            spec,
            upstream_root,
            data_root,
            run_root,
            max_steps=10_000,
            seed=0,
        ),
    ]
    print(
        "WSTS_PROTOTYPE_CONFIG="
        + json.dumps(
            {
                "prototype_id": BLOCK_PROTOTYPE_ID,
                "experiment_id": "C00",
                "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
                "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
                "block_fractions": [0.25, 0.50],
                "training_only": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    runpy.run_path(str(train_path), run_name="__main__")

    import torch

    print(
        f"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
