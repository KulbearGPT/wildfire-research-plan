"""Seal one declared 10K clean run after validating its local evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .matrix import CLEAN_RUNS, run_spec


MAX_STEPS = 10_000
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CheckpointLoader = Callable[[Path], Mapping[str, Any]]


def run_identity(run_id: str) -> tuple[str, int]:
    """Resolve the experiment and seed for one declared 10K clean run."""

    try:
        spec = run_spec(run_id)
    except ValueError as error:
        raise ValueError(f"run is not a declared 10K clean run: {run_id}") from error
    if spec.max_steps != MAX_STEPS or spec.stage == "screening":
        raise ValueError(f"run is not a declared 10K clean run: {run_id}")
    return spec.experiment_id, spec.seed


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


def finalize_run(
    run_id: str,
    run_root: Path,
    *,
    started_at_epoch: int,
    slurm_job_id: str,
    now_epoch: int | None = None,
    checkpoint_loader: CheckpointLoader = _load_checkpoint,
) -> dict[str, object]:
    """Validate training output and create one immutable completion record."""

    experiment, seed = run_identity(run_id)
    root = Path(run_root).resolve(strict=True)
    text = (root / "training.log").read_text(encoding="utf-8", errors="replace")
    marker = f"`Trainer.fit` stopped: `max_steps={MAX_STEPS}` reached."
    if marker not in text:
        raise ValueError(f"{MAX_STEPS}-step completion marker is missing")

    peak_matches = re.findall(r"WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=(\d+)", text)
    if not peak_matches or int(peak_matches[-1]) <= 0:
        raise ValueError("positive CUDA peak allocation marker is missing")

    checkpoints = tuple(sorted((root / "work").rglob("*.ckpt")))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one best checkpoint, got {len(checkpoints)}")
    checkpoint_payload = checkpoint_loader(checkpoints[0])
    checkpoint_step = checkpoint_payload.get("global_step", -1)
    if type(checkpoint_step) is not int or not 0 < checkpoint_step <= MAX_STEPS:
        raise ValueError(f"unexpected checkpoint global step: {checkpoint_step}")

    metrics = {
        "val_avg_precision": last_metric(text, "val_avg_precision"),
        "val_f1": last_metric(text, "val_f1"),
        "val_loss": last_metric(text, "val_loss"),
    }
    if not 0.0 <= metrics["val_avg_precision"] <= 1.0:
        raise ValueError("validation AP is outside [0, 1]")
    if not 0.0 <= metrics["val_f1"] <= 1.0:
        raise ValueError("validation F1 is outside [0, 1]")

    finished = int(time.time()) if now_epoch is None else now_epoch
    if type(started_at_epoch) is not int or finished < started_at_epoch:
        raise ValueError("run timing is invalid")
    if not slurm_job_id:
        raise ValueError("slurm_job_id must be nonempty")

    record: dict[str, object] = {
        "status": "pass",
        "purpose": "10000-step clean run; not a test-performance claim",
        "experiment": experiment,
        "slurm_job_id": slurm_job_id,
        "max_steps": MAX_STEPS,
        "seed": seed,
        "train_years": [2016, 2017, 2018, 2019, 2020],
        "validation_years": [2021],
        "test_enabled": False,
        "withheld_years": [2022, 2023],
        "checkpoint": str(checkpoints[0]),
        "checkpoint_global_step": checkpoint_step,
        "peak_cuda_allocated_bytes": int(peak_matches[-1]),
        "wall_seconds": finished - started_at_epoch,
        "metrics": metrics,
    }
    output = root / "completed.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    declared_10k = tuple(
        run_id for run_id, spec in CLEAN_RUNS.items() if spec.max_steps == MAX_STEPS
    )
    parser.add_argument("--run-id", choices=declared_10k, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--started-at-epoch", type=int, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record = finalize_run(
        args.run_id,
        args.run_root,
        started_at_epoch=args.started_at_epoch,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
