"""Seal one corrected-index 3K baseline after checking local evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .corrected_baselines import CORRECTED_BASELINES, corrected_baseline_spec


MAX_STEPS = 3_000
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CheckpointLoader = Callable[[Path], Mapping[str, Any]]


def last_metric(text: str, name: str) -> float:
    """Return the last finite value logged for a named validation metric."""

    normalized = ANSI.sub("", text).replace("\r", "\n")
    values: list[float] = []
    for line in normalized.splitlines():
        match = re.search(rf"\b{re.escape(name)}\b", line)
        if match is None:
            continue
        numeric = NUMBER.search(line[match.end() :])
        if numeric is not None:
            values.append(float(numeric.group(0)))
    if not values:
        raise ValueError(f"final validation metric is missing: {name}")
    value = values[-1]
    if not math.isfinite(value):
        raise ValueError(f"final validation metric is non-finite: {name}={value}")
    return value


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def finalize_corrected_baseline(
    baseline_id: str,
    run_root: Path,
    *,
    started_at_epoch: int,
    slurm_job_id: str,
    now_epoch: int | None = None,
    checkpoint_loader: CheckpointLoader = _load_checkpoint,
) -> dict[str, object]:
    """Validate a completed baseline and write its immutable record."""

    baseline = corrected_baseline_spec(baseline_id)
    root = Path(run_root).resolve(strict=True)
    output = root / "completed.json"
    if output.exists():
        raise FileExistsError(output)
    text = (root / "training.log").read_text(encoding="utf-8", errors="replace")
    marker = f"`Trainer.fit` stopped: `max_steps={MAX_STEPS}` reached."
    if marker not in text:
        raise ValueError("3000-step completion marker is missing")

    peak_matches = re.findall(r"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)", text)
    if not peak_matches or int(peak_matches[-1]) <= 0:
        raise ValueError("positive CUDA peak allocation marker is missing")
    checkpoints = tuple(sorted((root / "work").rglob("*.ckpt")))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one best checkpoint, got {len(checkpoints)}")
    checkpoint_step = checkpoint_loader(checkpoints[0]).get("global_step", -1)
    if type(checkpoint_step) is not int or not 0 < checkpoint_step <= MAX_STEPS:
        raise ValueError(f"unexpected checkpoint global step: {checkpoint_step}")

    finished = int(time.time()) if now_epoch is None else now_epoch
    if type(started_at_epoch) is not int or finished < started_at_epoch:
        raise ValueError("run timing is invalid")
    if not slurm_job_id:
        raise ValueError("slurm_job_id must be nonempty")
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "corrected-index 2021 baseline screen",
        "baseline_id": baseline.baseline_id,
        "experiment": baseline.experiment_id,
        "training_policy": baseline.training_policy,
        "seed": baseline.seed,
        "max_steps": baseline.max_steps,
        "corrected_index": True,
        "initialization": "from_scratch",
        "train_years": [2016, 2017, 2018, 2019, 2020],
        "validation_years": [2021],
        "test_enabled": False,
        "withheld_years": [2022, 2023],
        "slurm_job_id": slurm_job_id,
        "checkpoint": str(checkpoints[0]),
        "checkpoint_global_step": checkpoint_step,
        "peak_cuda_allocated_bytes": int(peak_matches[-1]),
        "wall_seconds": finished - started_at_epoch,
        "metrics": {
            "val_avg_precision": last_metric(text, "val_avg_precision"),
            "val_f1": last_metric(text, "val_f1"),
            "val_loss": last_metric(text, "val_loss"),
        },
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-id", choices=tuple(CORRECTED_BASELINES), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--started-at-epoch", type=int, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args(argv)
    record = finalize_corrected_baseline(
        args.baseline_id,
        args.run_root,
        started_at_epoch=args.started_at_epoch,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
