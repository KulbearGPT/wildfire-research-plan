"""Evaluate one immutable checkpoint/scenario/year task without retraining."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .evaluation import build_controlled_dataset
from .matrix import CORRUPTIONS, run_spec


@dataclass(frozen=True)
class SelectedTask:
    task: dict[str, object]
    mode: str
    heldout_authorized: bool


def select_task(
    manifest: Mapping[str, object], evaluation_id: str
) -> SelectedTask:
    """Select exactly one task after enforcing its engineering/formal boundary."""

    if manifest.get("schema_version") != 1 or manifest.get(
        "corruption_schema_version"
    ) != 1:
        raise ValueError("missingness manifest schema is unsupported")
    mode = manifest.get("mode")
    if mode not in {"engineering", "formal"}:
        raise ValueError("missingness manifest mode is invalid")
    records = manifest.get("records")
    tasks = manifest.get("tasks")
    if not isinstance(records, list) or not isinstance(tasks, list):
        raise ValueError("missingness manifest records and tasks must be lists")
    matches = [
        task
        for task in tasks
        if isinstance(task, dict) and task.get("evaluation_id") == evaluation_id
    ]
    if not matches:
        raise ValueError(f"evaluation task was not found: {evaluation_id}")
    if len(matches) != 1:
        raise ValueError("evaluation task must occur exactly once")
    task = dict(matches[0])

    required_task_keys = {
        "evaluation_id",
        "run_id",
        "experiment_id",
        "seed",
        "checkpoint",
        "scenario_id",
        "matrix_seed",
        "year",
        "output",
    }
    if set(task) != required_task_keys:
        raise ValueError("evaluation task fields differ from the required schema")
    run = run_spec(str(task["run_id"]))
    if run.max_steps != 10_000:
        raise ValueError("missingness evaluation requires a clean 10K run")
    if task["experiment_id"] != run.experiment_id or task["seed"] != run.seed:
        raise ValueError("evaluation task run identity is inconsistent")
    scenario_id = task["scenario_id"]
    if scenario_id not in CORRUPTIONS:
        raise ValueError("evaluation task scenario is unknown")
    if task["matrix_seed"] != CORRUPTIONS[str(scenario_id)].matrix_seed:
        raise ValueError("evaluation task matrix seed is inconsistent")

    matching_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("run_id") == task["run_id"]
    ]
    if len(matching_records) != 1:
        raise ValueError("evaluation task must have exactly one checkpoint record")
    if matching_records[0].get("checkpoint") != task["checkpoint"]:
        raise ValueError("evaluation checkpoint differs from its clean record")

    year = task["year"]
    if mode == "engineering":
        if (
            manifest.get("scientific_claim") is not False
            or manifest.get("heldout_access") is not False
            or manifest.get("years") != [2021]
            or year != 2021
        ):
            raise ValueError("engineering evaluation must remain on 2021")
        heldout_authorized = False
    else:
        record_ids = {
            record.get("run_id") for record in records if isinstance(record, dict)
        }
        if len(records) != 6 or len(record_ids) != 6:
            raise ValueError("formal evaluation requires six unique clean records")
        if (
            manifest.get("scientific_claim") is not True
            or manifest.get("heldout_access") is not True
            or manifest.get("years") != [2022, 2023]
            or year not in {2022, 2023}
        ):
            raise ValueError("formal evaluation boundary is invalid")
        heldout_authorized = True
    return SelectedTask(task, str(mode), heldout_authorized)


def evaluate_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, int | float]:
    """Compute exact AP and streaming threshold metrics for one data population."""

    model = model.to(device)
    model.eval()
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    true_positive = false_positive = false_negative = 0
    sample_count = pixel_count = 0
    weighted_loss = 0.0
    squared_probability_error = 0.0
    predictive_variance_total = 0.0
    has_predictive_variance = False

    with torch.inference_mode():
        for x, target in batches:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            uncertainty_forward = getattr(model, "forward_with_uncertainty", None)
            if callable(uncertainty_forward):
                logits, predictive_variance = uncertainty_forward(x)
                has_predictive_variance = True
            else:
                logits = model(x)
                predictive_variance = None
            if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            if (
                predictive_variance is not None
                and predictive_variance.ndim == target.ndim + 1
                and predictive_variance.shape[1] == 1
            ):
                predictive_variance = predictive_variance.squeeze(1)
            if logits.shape != target.shape:
                raise ValueError(
                    f"model output shape {tuple(logits.shape)} differs from target "
                    f"{tuple(target.shape)}"
                )
            if (
                predictive_variance is not None
                and predictive_variance.shape != target.shape
            ):
                raise ValueError("predictive variance shape differs from target")
            loss = model.compute_loss(logits, target)
            pixels = target.numel()
            weighted_loss += float(loss.detach().cpu()) * pixels
            probabilities = torch.sigmoid(logits)
            squared_probability_error += float(
                (probabilities - target.float()).square().sum().detach().cpu()
            )
            if predictive_variance is not None:
                predictive_variance_total += float(
                    predictive_variance.sum().detach().cpu()
                )
            pixel_count += pixels
            sample_count += target.shape[0]

            score_chunks.append(logits.detach().float().cpu().flatten())
            target_chunks.append(target.detach().bool().cpu().flatten())
            prediction = probabilities >= 0.5
            positive = target.bool()
            true_positive += int((prediction & positive).sum().item())
            false_positive += int((prediction & ~positive).sum().item())
            false_negative += int((~prediction & positive).sum().item())

    if sample_count == 0 or pixel_count == 0:
        raise ValueError("evaluation dataloader produced no samples")
    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].to(torch.float64)
    positive_count = int(sorted_targets.sum().item())
    if positive_count:
        cumulative_positive = torch.cumsum(sorted_targets, dim=0)
        threshold_ends = torch.ones_like(sorted_scores, dtype=torch.bool)
        threshold_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        end_indices = torch.nonzero(threshold_ends, as_tuple=False).flatten()
        true_positive_at_threshold = cumulative_positive[end_indices]
        precision_at_threshold = true_positive_at_threshold / (end_indices + 1)
        recall_at_threshold = true_positive_at_threshold / positive_count
        previous_recall = torch.cat(
            [torch.zeros(1, dtype=torch.float64), recall_at_threshold[:-1]]
        )
        ap = float(
            ((recall_at_threshold - previous_recall) * precision_at_threshold)
            .sum()
            .item()
        )
    else:
        ap = 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    metrics: dict[str, int | float] = {
        "sample_count": sample_count,
        "pixel_count": pixel_count,
        "avg_precision": ap,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "iou": true_positive / iou_denominator if iou_denominator else 0.0,
        "precision": precision,
        "recall": recall,
        "loss": weighted_loss / pixel_count,
        "brier": squared_probability_error / pixel_count,
    }
    if has_predictive_variance:
        metrics["mean_predictive_variance"] = (
            predictive_variance_total / pixel_count
        )
    return metrics


def checkpoint_init_args(
    hyperparameters: Mapping[str, Any],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Build constructor arguments compatible with the declared architecture."""

    init_args = dict(hyperparameters)
    init_args["encoder_weights"] = None
    if experiment_id == "C02":
        init_args.pop("use_doy", None)
    return init_args


def load_checkpoint_model(
    checkpoint: Path,
    *,
    experiment_id: str,
    upstream_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Strict-load the declared C00/C02 architecture from a Lightning checkpoint."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    module_name = "models.SMPModel" if experiment_id == "C00" else "models.SMPTempModel"
    class_name = "SMPModel" if experiment_id == "C00" else "SMPTempModel"
    model_class = getattr(importlib.import_module(module_name), class_name)

    payload = torch.load(
        Path(checkpoint).resolve(strict=True), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    hyperparameters = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint lacks hyperparameters or state_dict")
    init_args = checkpoint_init_args(
        hyperparameters,
        experiment_id=experiment_id,
    )
    model = model_class(**init_args)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def write_result_new(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("missingness manifest must be a JSON object")
    selected = select_task(manifest, args.evaluation_id)
    task = selected.task
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")

    dataset = build_controlled_dataset(
        upstream_root=args.upstream_root,
        data_root=args.data_root,
        stats_path=args.stats_path,
        experiment_id=str(task["experiment_id"]),
        scenario_id=str(task["scenario_id"]),
        evaluation_year=int(task["year"]),
        heldout_authorized=selected.heldout_authorized,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_checkpoint_model(
        Path(str(task["checkpoint"])),
        experiment_id=str(task["experiment_id"]),
        upstream_root=args.upstream_root,
        device=device,
    )
    metrics = evaluate_batches(model, loader, device=device)
    result = {
        "schema_version": 1,
        "status": "pass",
        "mode": selected.mode,
        "scientific_claim": selected.mode == "formal",
        "manifest": str(manifest_path),
        "task": task,
        "metrics": metrics,
        "boundary": {
            "is_train": False,
            "retraining": False,
            "checkpoint_selection": False,
            "natural_missingness_claim": False,
        },
    }
    write_result_new(Path(str(task["output"])), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
