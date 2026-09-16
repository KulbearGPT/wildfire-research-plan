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
    # Capture completed fit state before the subsequent best-checkpoint
    # validation can change trainer state. Publish only after the full upstream
    # command returns successfully; a saved best checkpoint alone is not proof
    # that all training steps ran.
    import pytorch_lightning as pl
    from .complete_baseline import sha256

    completed_fit = {}
    original_fit = pl.Trainer.fit

    def observed_fit(trainer, *fit_args, **fit_kwargs):
        result = original_fit(trainer, *fit_args, **fit_kwargs)
        callback = trainer.checkpoint_callback
        if trainer.global_step != baseline.max_steps:
            raise ValueError("baseline fit did not reach its required max_steps")
        if callback is None or callback.monitor != "val_avg_precision" or callback.mode != "max":
            raise ValueError("baseline requires best validation AP checkpoint selection")
        checkpoint = Path(callback.best_model_path).resolve(strict=True)
        completed_fit.update(
            fit_global_step=trainer.global_step,
            selection={
                "checkpoint": str(checkpoint.relative_to(run_root)),
                "monitor": callback.monitor,
                "mode": callback.mode,
                "score": float(callback.best_model_score),
                "sha256": sha256(checkpoint),
            },
        )
        return result

    pl.Trainer.fit = observed_fit
    try:
        runpy.run_path(str(train_path), run_name="__main__")
    finally:
        pl.Trainer.fit = original_fit
    if not completed_fit:
        raise ValueError("upstream command did not complete baseline fitting")
    receipt = {
        "schema_version": 1, "status": "pass",
        "baseline_id": baseline.baseline_id,
        "experiment": baseline.experiment_id,
        "training_policy": baseline.training_policy,
        "seed": baseline.seed, "max_steps": baseline.max_steps,
        "corrected_index": True, "initialization": "from_scratch",
        "validation_years": [2021], "test_enabled": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        **completed_fit,
    }
    with (run_root / "training-completion.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")

    import torch

    print(
        f"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
