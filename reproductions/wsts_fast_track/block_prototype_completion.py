"""Seal the one P02 FireDrop-plus-BlockDrop training run."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

from .completion import MAX_STEPS, _load_checkpoint, last_metric
from .prototype import (
    BLOCK_DROPOUT_PROBABILITY,
    BLOCK_PROTOTYPE_ID,
    FIRE_DROPOUT_PROBABILITY,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--started-at-epoch", type=int, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args(argv)

    root = args.run_root.resolve(strict=True)
    text = (root / "training.log").read_text(encoding="utf-8", errors="replace")
    if f"`Trainer.fit` stopped: `max_steps={MAX_STEPS}` reached." not in text:
        raise ValueError("10000-step completion marker is missing")
    peak_matches = re.findall(r"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)", text)
    if not peak_matches or int(peak_matches[-1]) <= 0:
        raise ValueError("positive CUDA peak allocation marker is missing")
    checkpoints = tuple(sorted((root / "work").rglob("*.ckpt")))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one best checkpoint, got {len(checkpoints)}")
    checkpoint_step = _load_checkpoint(checkpoints[0]).get("global_step", -1)
    if type(checkpoint_step) is not int or not 0 < checkpoint_step <= MAX_STEPS:
        raise ValueError(f"unexpected checkpoint global step: {checkpoint_step}")

    record = {
        "status": "pass",
        "purpose": "rapid 2021-selected block-robustness prototype",
        "prototype_id": BLOCK_PROTOTYPE_ID,
        "experiment": "C00",
        "seed": 0,
        "max_steps": MAX_STEPS,
        "training_policy": {
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
            "block_fractions": [0.25, 0.50],
            "training_only": True,
        },
        "train_years": [2016, 2017, 2018, 2019, 2020],
        "validation_years": [2021],
        "test_enabled": False,
        "withheld_years": [2022, 2023],
        "slurm_job_id": args.slurm_job_id,
        "checkpoint": str(checkpoints[0]),
        "checkpoint_global_step": checkpoint_step,
        "peak_cuda_allocated_bytes": int(peak_matches[-1]),
        "wall_seconds": int(time.time()) - args.started_at_epoch,
        "metrics": {
            "val_avg_precision": last_metric(text, "val_avg_precision"),
            "val_f1": last_metric(text, "val_f1"),
            "val_loss": last_metric(text, "val_loss"),
        },
    }
    output = root / "completed.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
