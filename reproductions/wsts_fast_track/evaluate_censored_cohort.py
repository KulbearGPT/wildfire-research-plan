"""Compare matched legacy and target-censored fine-tunes on the 2021 QA gate."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .censored_training import PROTOTYPE_IDS
from .contract import validate_inventory
from .entrypoint import _install_runtime_contract, load_training_stats
from .environment_dro import resolve_dataset_index
from .evaluate_belief_state import LatestDayP00, _build_evaluation_dataset
from .evaluate_missingness import load_checkpoint_model, write_result_new
from .evaluate_spatial_router import _checkpoint_from_record
from .evaluate_viirs_reliability import NaturalReliabilityDataset
from .evaluate_viirs_target_quality import (
    TargetQualityDataset,
    binary_metrics,
    target_observation_mask,
)
from .prototype import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID


def _load_candidate(
    path: Path,
    objective: str,
    *,
    p00_record: Path,
    p00_checkpoint: Path,
    upstream: Path,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "pass"
        or payload.get("prototype_id") != PROTOTYPE_IDS[objective]
        or payload.get("objective") != objective
        or Path(str(payload.get("base_p00_record", ""))).resolve() != p00_record
        or Path(str(payload.get("base_p00_checkpoint", ""))).resolve() != p00_checkpoint
        or not isinstance(payload.get("model_state"), dict)
    ):
        raise ValueError(f"invalid matched {objective} cohort checkpoint")
    model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return LatestDayP00(model).to(device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--censored-checkpoint", type=Path, required=True)
    parser.add_argument("--input-reliability-root", type=Path, required=True)
    parser.add_argument("--target-reliability-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    dataset_class = importlib.import_module("dataloader.FireSpreadDataset").FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    standard = _build_evaluation_dataset(
        dataset_class, data_root=data_root, history=1, scenario_id="M00"
    )
    dataset = TargetQualityDataset(
        NaturalReliabilityDataset(standard, args.input_reliability_root),
        args.target_reliability_root,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    p00_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    models = {
        "p00": LatestDayP00(p00_model).to(device),
        "legacy": _load_candidate(
            args.legacy_checkpoint,
            "legacy",
            p00_record=p00_record,
            p00_checkpoint=p00_checkpoint,
            upstream=upstream,
            device=device,
        ),
        "censored": _load_candidate(
            args.censored_checkpoint,
            "censored",
            p00_record=p00_record,
            p00_checkpoint=p00_checkpoint,
            upstream=upstream,
            device=device,
        ),
    }
    logits_by_model: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    target_chunks: list[torch.Tensor] = []
    reliability_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        for packed, target, reliability, sample_index in loader:
            packed = packed.to(device, non_blocking=True)
            for name, model in models.items():
                logits = model(packed)
                if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                if logits.shape != target.shape:
                    raise ValueError("model output shape differs from target")
                logits_by_model[name].append(logits.float().cpu())
            target_chunks.append(target.long().cpu())
            reliability_chunks.append(reliability.to(torch.uint8).cpu())
            index_chunks.append(sample_index.long().cpu())
    targets = torch.cat(target_chunks)
    reliability = torch.cat(reliability_chunks)
    indices = torch.cat(index_chunks)
    if indices.tolist() != list(range(len(dataset))):
        raise ValueError("2021 target-quality population is incomplete")
    include = target_observation_mask(targets, reliability)
    metrics = {
        name: {
            "standard": binary_metrics(torch.cat(chunks), targets),
            "qa_censored": binary_metrics(torch.cat(chunks), targets, include),
        }
        for name, chunks in logits_by_model.items()
    }

    legacy = metrics["legacy"]
    censored = metrics["censored"]
    comparison = {
        "qa_ap_delta": float(censored["qa_censored"]["avg_precision"])
        - float(legacy["qa_censored"]["avg_precision"]),
        "qa_brier_delta": float(censored["qa_censored"]["brier"])
        - float(legacy["qa_censored"]["brier"]),
        "qa_loss_delta": float(censored["qa_censored"]["loss"])
        - float(legacy["qa_censored"]["loss"]),
        "standard_ap_delta": float(censored["standard"]["avg_precision"])
        - float(legacy["standard"]["avg_precision"]),
    }
    directional_pass = (
        comparison["qa_ap_delta"] > 0
        and comparison["qa_brier_delta"] < 0
        and comparison["qa_loss_delta"] < 0
        and comparison["standard_ap_delta"] >= 0
    )
    material = comparison["qa_ap_delta"] >= 1e-3
    promotion = directional_pass and material
    verdict = "promote" if promotion else "null" if directional_pass else "reject"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "scientific_claim": False,
        "purpose": "matched 3K target-censoring feasibility experiment",
        "year": 2021,
        "sample_count": len(dataset),
        "target_mask_policy": "positive-or-reliably-observed",
        "material_ap_threshold": 1e-3,
        "p00_record": str(p00_record),
        "legacy_checkpoint": str(args.legacy_checkpoint.resolve(strict=True)),
        "censored_checkpoint": str(args.censored_checkpoint.resolve(strict=True)),
        "label_population": {
            "total_pixels": int(targets.numel()),
            "positive_pixels": int(targets.bool().sum().item()),
            "qa_included_pixels": int(include.sum().item()),
            "unknown_zero_pixels": int(((~targets.bool()) & (~reliability.bool())).sum().item()),
        },
        "models": metrics,
        "comparison_censored_minus_legacy": comparison,
        "promotion": {
            "directional_pass": directional_pass,
            "material_qa_ap": material,
            "pass": promotion,
            "verdict": verdict,
        },
    }
    write_result_new(output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
