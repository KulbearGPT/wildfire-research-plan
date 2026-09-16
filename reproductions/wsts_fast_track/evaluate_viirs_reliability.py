"""Run a 24-sample T=1 attention screen with natural VIIRS reliability."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import validate_inventory
from .runtime import _install_runtime_contract, load_training_stats
from .diagnostic_support import resolve_dataset_index
from .evaluate_belief_state import (
    LatestDayP00,
    _build_evaluation_dataset,
    _load_validated_checkpoint,
)
from .evaluate_missingness import evaluate_batches, load_checkpoint_model, write_result_new
from .diagnostic_support import _checkpoint_from_record
from .evaluation import _center_crop_last_two
from .latent_state_common import load_trainable_state_dict
from .diagnostic_support import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .train_belief_state import build_model
from .viirs_reliability import FIXED_2021_MODEL_GATE


def first_event_indices(base: Any, event_names: Sequence[str]) -> dict[str, int]:
    """Find the first common-population sample for each requested 2021 event."""
    requested = set(event_names)
    if len(requested) != len(event_names):
        raise ValueError("event names must be unique")
    result: dict[str, int] = {}
    for index in range(len(base)):
        year, fire_name, in_fire_index = base.find_image_index_from_dataset_index(index)
        if year == 2021 and fire_name in requested and in_fire_index == 0:
            if fire_name in result:
                raise ValueError(f"duplicate first sample for {fire_name}")
            result[fire_name] = index
    missing = requested - set(result)
    if missing:
        raise ValueError(f"missing first samples for events: {sorted(missing)}")
    return result


def replace_t1_reliability(
    packed: torch.Tensor, full_reliability: np.ndarray
) -> torch.Tensor:
    """Replace only channel 40 with the center-cropped natural reliability map."""
    if packed.ndim != 4 or packed.shape[0] != 1 or packed.shape[1] != 41:
        raise ValueError("T=1 packed sample must have shape [1,41,H,W]")
    reliability = np.asarray(full_reliability)
    if reliability.ndim != 2 or not set(np.unique(reliability)).issubset({0, 1}):
        raise ValueError("natural reliability must be a two-dimensional binary map")
    if packed.shape[-2] != packed.shape[-1]:
        raise ValueError("natural reliability screen requires a square model crop")
    crop = _center_crop_last_two(reliability, packed.shape[-1])
    result = packed.clone()
    result[0, 40] = torch.as_tensor(crop, dtype=packed.dtype, device=packed.device)
    return result


class NaturalReliabilityDataset(Dataset[Any]):
    """Select the fixed model gate and replace synthetic M00 validity with VIIRS R."""

    def __init__(self, standard: Any, reliability_root: Path) -> None:
        if standard.controlled.base.n_leading_observations != 1:
            raise ValueError("natural reliability mini-screen is T=1 only")
        self.standard = standard
        self.root = Path(reliability_root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        records = manifest.get("samples")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("gate") != "model"
            or not isinstance(records, list)
            or len(records) != len(FIXED_2021_MODEL_GATE)
        ):
            raise ValueError("natural reliability manifest is not the fixed model gate")
        by_identity = {
            (record.get("event"), record.get("day")): record
            for record in records
            if isinstance(record, dict)
        }
        expected = set(FIXED_2021_MODEL_GATE)
        if set(by_identity) != expected:
            raise ValueError("natural reliability identities differ from the fixed model gate")
        indices = first_event_indices(
            standard.controlled.base, tuple(event for event, _day in FIXED_2021_MODEL_GATE)
        )
        self.samples = tuple(
            (event, day, indices[event], by_identity[(event, day)])
            for event, day in FIXED_2021_MODEL_GATE
        )
        self._reliability_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        event, day, base_index, record = self.samples[index]
        packed, target = self.standard[base_index]
        relative = str(record["output"])
        if relative not in self._reliability_cache:
            path = self.root / relative
            with np.load(path) as payload:
                self._reliability_cache[relative] = payload["reliability"].copy()
        natural = self._reliability_cache[relative]
        return replace_t1_reliability(packed, natural), target

    def mean_reliable_fraction(self) -> float:
        return float(
            sum(float(record["reliable_fraction"]) for _, _, _, record in self.samples)
            / len(self.samples)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--artifact-map", type=Path, help="checksum-verified relocation map for historical artifact paths")
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--reliability-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        artifact_map=args.artifact_map,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    attention_checkpoint = args.attention_checkpoint.resolve(strict=True)
    checkpoint = _load_validated_checkpoint(
        attention_checkpoint,
        p00_record=p00_record,
        p00_checkpoint=p00_checkpoint,
        artifact_map=args.artifact_map,
    )
    if checkpoint["method"] != "attention" or checkpoint["history"] != 1:
        raise ValueError("natural reliability mini-screen requires attention T=1")
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
    standard = _build_evaluation_dataset(
        dataset_class, data_root=data_root, history=1, scenario_id="M00"
    )
    dataset = NaturalReliabilityDataset(standard, args.reliability_root)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    default_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    attention = build_model("attention", default_model, 1).to(device)
    load_trainable_state_dict(attention, checkpoint["model_state"])
    if (
        sum(parameter.numel() for parameter in attention.trainable_parameters())
        != checkpoint["trainable_parameters"]
    ):
        raise ValueError("attention parameter count differs from checkpoint")
    p00 = LatestDayP00(default_model).to(device)

    attention_metrics = evaluate_batches(attention, loader, device=device)
    p00_metrics = evaluate_batches(p00, loader, device=device)
    result = {
        "schema_version": 1,
        "status": "pass",
        "scientific_claim": False,
        "purpose": "fixed 24-sample natural-reliability mini-screen",
        "year": 2021,
        "history": 1,
        "sample_count": len(dataset),
        "mean_reliable_fraction": dataset.mean_reliable_fraction(),
        "age_used": False,
        "attention_checkpoint": str(attention_checkpoint),
        "p00_record": str(p00_record),
        "attention": {"metrics": attention_metrics},
        "p00": {"metrics": p00_metrics},
        "ap_delta": float(attention_metrics["avg_precision"])
        - float(p00_metrics["avg_precision"]),
        "brier_delta": float(attention_metrics["brier"])
        - float(p00_metrics["brier"]),
    }
    write_result_new(output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
