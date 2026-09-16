"""Evaluate one fire belief-state prototype against its frozen P00 base."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .diagnostic_artifacts import provenance_matches
from .contract import validate_inventory
from .runtime import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .diagnostic_support import resolve_dataset_index
from .evaluate_missingness import evaluate_batches, load_checkpoint_model, write_result_new
from .diagnostic_support import _checkpoint_from_record
from .evaluation import ControlledMissingnessDataset
from .latent_state_common import (
    SCENARIOS,
    TRAIN_SCENARIOS,
    BeliefStateEvaluationDataset,
    load_trainable_state_dict,
    masked_unweighted_focal,
    screen_2021,
    split_observations,
)
from .diagnostic_support import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .train_belief_state import (
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    METHODS,
    PROTOTYPE_ID,
    SEED,
    STATE_WEIGHT,
    TRAINING_STEPS,
    build_model,
)


screen_payload = screen_2021
EVALUATION_YEAR = 2021
_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "prototype_id",
    "method",
    "history",
    "p00_record",
    "p00_checkpoint",
    "git_commit",
    "train_years",
    "seed",
    "steps",
    "optimizer",
    "learning_rate",
    "physical_batch_size",
    "effective_batch_size",
    "accumulation_steps",
    "corruptions",
    "state_weight",
    "trainable_parameters",
    "final_loss",
    "final_loss_components",
    "model_state",
}


def selected_scenarios(scenario_id: str | None) -> tuple[str, ...]:
    """Select one isolated scenario or the complete frozen evaluation matrix."""

    if scenario_id is None:
        return SCENARIOS
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown belief-state scenario: {scenario_id}")
    return (scenario_id,)


class LatestDayP00(torch.nn.Module):
    """Forecast the corrupted latest 40-feature day with frozen P00."""

    def __init__(self, default_model: torch.nn.Module) -> None:
        super().__init__()
        self.default_model = default_model

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        features, _reliability = split_observations(packed)
        return self.default_model(features[:, -1:])

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return masked_unweighted_focal(logits, target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--belief-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--device", default="cuda")
    return parser


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load_validated_checkpoint(
    path: Path,
    *,
    p00_record: Path,
    p00_checkpoint: Path,
    artifact_map: Path | None = None,
) -> dict[str, Any]:
    payload = torch.load(
        path.resolve(strict=True), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("belief-state checkpoint fields differ from schema version one")
    components = payload.get("final_loss_components")
    model_state = payload.get("model_state")
    method = payload.get("method")
    history = payload.get("history")
    physical_batch = payload.get("physical_batch_size")
    accumulation = payload.get("accumulation_steps")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "pass"
        or payload.get("prototype_id") != PROTOTYPE_ID
        or method not in METHODS
        or type(history) is not int
        or history not in {1, 5}
        or not provenance_matches(payload.get("p00_record"), p00_record, artifact_map=artifact_map)
        or not provenance_matches(payload.get("p00_checkpoint"), p00_checkpoint, artifact_map=artifact_map)
        or not isinstance(payload.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["git_commit"]) is None
        or payload.get("train_years") != list(TRAIN_YEARS)
        or payload.get("seed") != SEED
        or payload.get("steps") != TRAINING_STEPS
        or payload.get("optimizer") != "AdamW"
        or payload.get("learning_rate") != LEARNING_RATE
        or type(physical_batch) is not int
        or physical_batch <= 0
        or type(accumulation) is not int
        or accumulation <= 0
        or physical_batch * accumulation != EFFECTIVE_BATCH_SIZE
        or payload.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE
        or payload.get("corruptions") != list(TRAIN_SCENARIOS)
        or payload.get("state_weight") != STATE_WEIGHT
        or type(payload.get("trainable_parameters")) is not int
        or payload["trainable_parameters"] <= 0
        or not _is_finite_number(payload.get("final_loss"))
        or not isinstance(components, dict)
        or set(components) != {"forecast", "state"}
        or not _is_finite_number(components.get("state"))
        or not isinstance(model_state, Mapping)
        or not model_state
    ):
        raise ValueError("invalid fire belief-state checkpoint metadata")
    if method == "reconstruction":
        if components.get("forecast") is not None:
            raise ValueError("reconstruction checkpoint must have null forecast loss")
    elif not _is_finite_number(components.get("forecast")):
        raise ValueError("task-oriented checkpoint requires finite forecast loss")
    for key, value in model_state.items():
        if not isinstance(key, str) or key.startswith("default_model."):
            raise ValueError("belief-state checkpoint contains invalid state keys")
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("belief-state checkpoint contains non-finite state")
    return payload


def _build_evaluation_dataset(
    dataset_class: type,
    *,
    data_root: Path,
    history: int,
    scenario_id: str,
) -> BeliefStateEvaluationDataset:
    base = dataset_class(
        data_dir=str(data_root),
        included_fire_years=[EVALUATION_YEAR],
        n_leading_observations=history,
        n_leading_observations_test_adjustment=6,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=False,
        remove_duplicate_features=False,
        features_to_keep=None,
        return_doy=False,
        stats_years=list(TRAIN_YEARS),
        is_pad=False,
    )
    controlled = ControlledMissingnessDataset(
        base,
        scenario_id,
        evaluation_year=EVALUATION_YEAR,
        heldout_authorized=False,
        routing_mask_channel=True,
    )
    return BeliefStateEvaluationDataset(controlled)


def _require_exact_m00(
    model: torch.nn.Module,
    p00: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> None:
    model.eval()
    p00.eval()
    checked_batches = 0
    with torch.inference_mode():
        for packed, _target in loader:
            packed = packed.to(device, non_blocking=True)
            model_logits = model(packed)
            p00_logits = p00(packed)
            if not torch.equal(model_logits, p00_logits):
                raise ValueError("M00 belief logits differ from P00")
            checked_batches += 1
    if checked_batches == 0 or checked_batches != len(loader):
        raise ValueError("M00 exactness check did not cover every batch")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    belief_checkpoint = args.belief_checkpoint.resolve(strict=True)
    checkpoint = _load_validated_checkpoint(
        belief_checkpoint,
        p00_record=p00_record,
        p00_checkpoint=p00_checkpoint,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    default_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    model = build_model(
        str(checkpoint["method"]), default_model, int(checkpoint["history"])
    ).to(device)
    load_trainable_state_dict(model, checkpoint["model_state"])
    if (
        sum(parameter.numel() for parameter in model.trainable_parameters())
        != checkpoint["trainable_parameters"]
    ):
        raise ValueError("belief-state parameter count differs from checkpoint")
    p00 = LatestDayP00(default_model).to(device)
    output_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, dict[str, Any]] = {}
    for scenario_id in selected_scenarios(args.scenario):
        dataset = _build_evaluation_dataset(
            dataset_class,
            data_root=data_root,
            history=int(checkpoint["history"]),
            scenario_id=scenario_id,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model_metrics = evaluate_batches(model, loader, device=device)
        p00_metrics = evaluate_batches(p00, loader, device=device)
        if scenario_id == "M00":
            _require_exact_m00(model, p00, loader, device=device)
        ap_delta = float(model_metrics["avg_precision"]) - float(
            p00_metrics["avg_precision"]
        )
        result = {
            "schema_version": 1,
            "status": "pass",
            "scientific_claim": False,
            "prototype_id": PROTOTYPE_ID,
            "method": checkpoint["method"],
            "history": checkpoint["history"],
            "scenario_id": scenario_id,
            "year": EVALUATION_YEAR,
            "model": {
                "prototype_id": PROTOTYPE_ID,
                "metrics": model_metrics,
            },
            "p00": {
                "prototype_id": P00_ID,
                "metrics": p00_metrics,
            },
            "ap_delta": ap_delta,
        }
        write_result_new(output_root / f"{scenario_id}.json", result)
        results[scenario_id] = result
        print(json.dumps(result, sort_keys=True), flush=True)

    if args.scenario is not None:
        return 0

    screened_results = {
        scenario_id: results[scenario_id] for scenario_id in TRAIN_SCENARIOS
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "scientific_claim": False,
        "prototype_id": PROTOTYPE_ID,
        "method": checkpoint["method"],
        "history": checkpoint["history"],
        "git_commit": checkpoint["git_commit"],
        "p00_record": str(p00_record),
        "p00_checkpoint": str(p00_checkpoint),
        "belief_checkpoint": str(belief_checkpoint),
        "year": EVALUATION_YEAR,
        "m00_exact": True,
        "results": results,
        "screen_2021": screen_2021(screened_results),
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
