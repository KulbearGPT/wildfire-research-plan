"""Replay archived P00 (invalid pooled-index foundation), not a corrected baseline."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from .contract import experiment_spec, upstream_arguments, validate_inventory
from .runtime import _install_runtime_contract, load_training_stats
from .corruption_training import (
    FIRE_DROPOUT_PROBABILITY,
    install_training_fire_dropout,
)


PROTOTYPE_ID = "P00-FireDrop-C00"
MAX_STEPS = 10_000

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("legacy P00 replay requires a Slurm allocation")
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
    # Preserve the pinned upstream index implementation, including its known
    # pooled-year loop leak. Never install the corrected baseline resolver here.
    import importlib
    dataset_class = importlib.import_module("dataloader.FireSpreadDataset").FireSpreadDataset
    if dataset_class.find_image_index_from_dataset_index.__module__ != "dataloader.FireSpreadDataset":
        raise ValueError("legacy P00 requires an untouched upstream index resolver")
    install_training_fire_dropout(upstream_root)

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
            max_steps=MAX_STEPS,
            seed=0,
        ),
    ]
    print(
        "WSTS_PROTOTYPE_CONFIG="
        + json.dumps(
            {
                "prototype_id": PROTOTYPE_ID,
                "experiment_id": "C00",
                "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
                "training_only": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    import pytorch_lightning as pl
    from .complete_baseline import sha256

    original_fit = pl.Trainer.fit
    completed_fit = {}

    def observed_fit(trainer, *fit_args, **fit_kwargs):
        result = original_fit(trainer, *fit_args, **fit_kwargs)
        callback = trainer.checkpoint_callback
        if trainer.global_step != MAX_STEPS:
            raise ValueError("legacy P00 did not finish its full fit budget")
        if callback is None or callback.monitor != "val_avg_precision" or callback.mode != "max":
            raise ValueError("legacy P00 requires best validation AP selection")
        checkpoint = Path(callback.best_model_path).resolve(strict=True)
        completed_fit.update(
            fit_global_step=trainer.global_step,
            checkpoint=str(checkpoint.relative_to(run_root)),
            checkpoint_sha256=sha256(checkpoint),
            selection_score=float(callback.best_model_score),
            selection_monitor=callback.monitor, selection_mode=callback.mode,
        )
        return result

    pl.Trainer.fit = observed_fit
    try:
        runpy.run_path(str(train_path), run_name="__main__")
    finally:
        pl.Trainer.fit = original_fit
    if not completed_fit:
        raise ValueError("upstream did not complete legacy P00 fitting")
    receipt = {
        "status": "pass" if MAX_STEPS == 10_000 else "qualification",
        "scientific_claim": False,
        "legacy_index_semantics": True,
        "prototype_id": PROTOTYPE_ID, "experiment": "C00", "seed": 0,
        "max_steps": MAX_STEPS,
        "training_policy": {"active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
                            "training_only": True},
        "validation_years": [2021], "test_enabled": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        **completed_fit,
    }
    with (run_root / "legacy-training-completion.json").open("x") as handle:
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
