"""Train one corrected-index B0--B3 baseline from scratch."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from .contract import experiment_spec, upstream_arguments, validate_inventory
from .corrected_baselines import (
    CORRECTED_BASELINES,
    corrected_baseline_spec,
    install_corrected_baseline,
)
from .runtime import _install_runtime_contract, load_training_stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-id", choices=tuple(CORRECTED_BASELINES), required=True
    )
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
    baseline = corrected_baseline_spec(args.baseline_id)
    experiment = experiment_spec(baseline.experiment_id)
    _install_runtime_contract(upstream_root, experiment.experiment_id, stats)
    install_corrected_baseline(upstream_root, baseline.baseline_id)

    work_root = run_root / "work"
    work_root.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(work_root)
    sys.argv = [
        str(train_path),
        *upstream_arguments(
            experiment,
            upstream_root,
            data_root,
            run_root,
            max_steps=baseline.max_steps,
            seed=baseline.seed,
        ),
    ]
    print(
        "WSTS_CORRECTED_BASELINE_CONFIG="
        + json.dumps(
            {
                "baseline_id": baseline.baseline_id,
                "experiment_id": baseline.experiment_id,
                "training_policy": baseline.training_policy,
                "corrected_index": True,
                "initialization": "from_scratch",
                "max_steps": baseline.max_steps,
                "seed": baseline.seed,
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
