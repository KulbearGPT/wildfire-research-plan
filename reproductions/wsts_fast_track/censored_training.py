"""Minimal primitives for the matched target-censoring prototype."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .latent_state_common import sigmoid_focal_loss


COHORT_YEARS = (2016, 2017, 2018, 2019, 2020)
COHORT_PER_YEAR = 8
TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-4
SEED = 0
PROTOTYPE_IDS = {
    "legacy": "P14-TargetQACohortLegacyFocal",
    "censored": "P15-TargetQACohortCensoredFocal",
}


def select_balanced_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    years: Sequence[int],
    per_year: int,
) -> list[dict[str, Any]]:
    """Select a deterministic event-unique half-positive cohort per year."""

    if per_year <= 0 or per_year % 2:
        raise ValueError("per-year cohort size must be positive and even")
    quota = per_year // 2
    selected: list[dict[str, Any]] = []
    for year in years:
        year_records = [dict(record) for record in candidates if record.get("year") == year]
        by_state = {
            positive: sorted(
                (
                    record
                    for record in year_records
                    if (int(record["target_positive_pixels"]) > 0) is positive
                ),
                key=lambda record: hashlib.sha256(
                    f"{record['year']}/{record['event']}/{record['target_day']}".encode()
                ).hexdigest(),
            )
            for positive in (True, False)
        }
        used_events: set[str] = set()
        for positive in (True, False):
            chosen: list[dict[str, Any]] = []
            for record in by_state[positive]:
                event = str(record["event"])
                if event in used_events:
                    continue
                chosen.append(record)
                used_events.add(event)
                if len(chosen) == quota:
                    break
            if len(chosen) != quota:
                state = "positive" if positive else "zero"
                raise ValueError(f"year {year} lacks {quota} event-unique {state} targets")
            selected.extend(chosen)
    return sorted(selected, key=lambda record: (int(record["year"]), str(record["event"])))


def preprocess_aligned(
    base: Any,
    features: np.ndarray,
    target: np.ndarray,
    reliability: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the upstream training geometry once to target and reliability."""

    if target.shape != reliability.shape:
        raise ValueError("target and reliability must have equal raw shapes")
    if not getattr(base, "is_train", False):
        raise ValueError("aligned preprocessing requires a training dataset")
    if getattr(base, "is_pad", False):
        raise ValueError("aligned preprocessing does not support padded models")

    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.stack(
        (
            torch.as_tensor(target > 0, dtype=torch.long),
            torch.as_tensor(reliability > 0, dtype=torch.long),
        )
    )
    side = int(base.crop_side_length)
    if x.shape[-2] <= side or x.shape[-1] <= side:
        raise ValueError("training images must be larger than the requested crop")

    best_score = -1.0
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    for _ in range(10):
        top = int(np.random.randint(0, x.shape[-2] - side))
        left = int(np.random.randint(0, x.shape[-1] - side))
        x_crop = x[..., top : top + side, left : left + side]
        y_crop = y[..., top : top + side, left : left + side]
        score = float(x_crop[:, -1].float().mean() + 1000 * y_crop[0].float().mean())
        if score > best_score:
            best_score = score
            best = (x_crop, y_crop)
    if best is None:
        raise RuntimeError("aligned crop selection produced no crop")
    x, y = best

    horizontal = bool(np.random.random() > 0.5)
    vertical = bool(np.random.random() > 0.5)
    quarter_turns = int(np.floor(np.random.random() * 4))
    if horizontal:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
        x[:, base.indices_of_degree_features] = 360 - x[:, base.indices_of_degree_features]
    if vertical:
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,))
        x[:, base.indices_of_degree_features] = (
            180 - x[:, base.indices_of_degree_features]
        ) % 360
    if quarter_turns:
        angle = quarter_turns * 90
        x = torch.rot90(x, k=quarter_turns, dims=(-2, -1))
        y = torch.rot90(y, k=quarter_turns, dims=(-2, -1))
        x[:, base.indices_of_degree_features] = (
            x[:, base.indices_of_degree_features] - angle
        ) % 360

    x[:, base.indices_of_degree_features] = torch.sin(
        torch.deg2rad(x[:, base.indices_of_degree_features])
    )
    binary_active_fire = (x[:, -1:] > 0).float()
    x = base.standardize_features(x)
    x = torch.cat((x, binary_active_fire), dim=1)
    x = torch.nan_to_num(x, nan=0.0)
    new_shape = (x.shape[0], x.shape[2], x.shape[3], base.one_hot_matrix.shape[0])
    landcover = x[:, 16].long().flatten() - 1
    encoded = base.one_hot_matrix[landcover].reshape(new_shape).permute(0, 3, 1, 2)
    x = torch.cat((x[:, :16], encoded, x[:, 17:]), dim=1)
    return x, y[0].long(), y[1].to(torch.uint8)


def cohort_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    *,
    censored: bool,
) -> torch.Tensor:
    """Compute the matched legacy or positive-or-observed focal objective."""

    if logits.shape != target.shape or target.shape != reliability.shape:
        raise ValueError("logits, target, and reliability must have equal shapes")
    losses = sigmoid_focal_loss(
        logits,
        target.to(dtype=logits.dtype),
        alpha=-1.0,
        gamma=2.0,
        reduction="none",
    )
    if not censored:
        return losses.mean()
    include = target.bool() | reliability.bool()
    selected = losses[include]
    if selected.numel() == 0:
        raise ValueError("censored cohort crop contains no supervised pixels")
    return selected.mean()


def load_cohort_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the frozen 40-event training cohort."""

    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    records = payload.get("samples") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != 1
        or payload.get("years") != list(COHORT_YEARS)
        or payload.get("per_year") != COHORT_PER_YEAR
        or not isinstance(records, list)
        or len(records) != len(COHORT_YEARS) * COHORT_PER_YEAR
    ):
        raise ValueError("training cohort manifest differs from the frozen 40-event schema")
    required = {
        "year",
        "event",
        "dataset_index",
        "in_fire_index",
        "input_day",
        "target_day",
        "target_positive_pixels",
        "hdf5",
    }
    if any(not isinstance(record, dict) or set(record) != required for record in records):
        raise ValueError("training cohort sample fields differ from schema version one")
    for year in COHORT_YEARS:
        subset = [record for record in records if record["year"] == year]
        positives = sum(int(record["target_positive_pixels"]) > 0 for record in subset)
        if len(subset) != COHORT_PER_YEAR or positives != COHORT_PER_YEAR // 2:
            raise ValueError(f"year {year} is not balanced positive/zero")
        if len({record["event"] for record in subset}) != COHORT_PER_YEAR:
            raise ValueError(f"year {year} contains repeated events")
    return [dict(record) for record in records]


class CensoredCohortDataset(Dataset[Any]):
    """Preload and augment the fixed cohort with aligned target reliability."""

    def __init__(self, base: Any, cohort_path: Path, reliability_root: Path) -> None:
        self.base = base
        self.records = load_cohort_manifest(cohort_path)
        reliability_root = reliability_root.resolve(strict=True)
        reliability_manifest = json.loads(
            (reliability_root / "manifest.json").read_text(encoding="utf-8")
        )
        qa_records = reliability_manifest.get("samples")
        if (
            reliability_manifest.get("schema_version") != 1
            or reliability_manifest.get("gate") != "training-cohort-target"
            or not isinstance(qa_records, list)
        ):
            raise ValueError("target reliability manifest is not a training cohort")
        qa_by_identity = {
            (record.get("year"), record.get("event"), record.get("day")): record
            for record in qa_records
            if isinstance(record, dict)
        }
        expected = {
            (record["year"], record["event"], record["target_day"])
            for record in self.records
        }
        if set(qa_by_identity) != expected:
            raise ValueError("target reliability identities differ from training cohort")

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for record in self.records:
            year = int(record["year"])
            event = str(record["event"])
            in_fire_index = int(record["in_fire_index"])
            features, target = base.load_imgs(year, event, in_fire_index)
            qa_record = qa_by_identity[(year, event, record["target_day"])]
            with np.load(reliability_root / str(qa_record["output"])) as payload:
                reliability = payload["reliability"].copy()
            if target.shape != reliability.shape:
                raise ValueError("cohort target and reliability raster shapes differ")
            self.samples.append((features, target, reliability))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, target, reliability = self.samples[index]
        processed, label, aligned_reliability = preprocess_aligned(
            self.base, features.copy(), target.copy(), reliability.copy()
        )
        if processed.shape[0] != 1:
            raise ValueError("censored cohort requires one input day")
        return processed, label, aligned_reliability
