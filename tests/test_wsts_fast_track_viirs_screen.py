from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
import torch


def test_first_event_indices_select_the_common_t1_t5_population() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_reliability import (
        first_event_indices,
    )

    class Base:
        rows = (
            (2021, "fire_a", 0),
            (2021, "fire_a", 1),
            (2021, "fire_b", 0),
            (2021, "fire_b", 1),
        )

        def __len__(self) -> int:
            return len(self.rows)

        def find_image_index_from_dataset_index(self, index: int):
            return self.rows[index]

    assert first_event_indices(Base(), ("fire_b", "fire_a")) == {
        "fire_a": 0,
        "fire_b": 2,
    }


def test_natural_reliability_replaces_only_the_packed_validity_crop() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_reliability import (
        replace_t1_reliability,
    )

    packed = torch.randn((1, 41, 4, 4))
    packed[:, 40] = 1.0
    original_features = packed[:, :40].clone()
    full = np.zeros((6, 6), dtype=np.uint8)
    full[1:5, 1:5] = np.eye(4, dtype=np.uint8)

    replaced = replace_t1_reliability(packed, full)

    assert torch.equal(replaced[:, :40], original_features)
    assert torch.equal(replaced[0, 40], torch.eye(4))


def test_target_observation_mask_keeps_positives_and_excludes_unknown_zeros() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_target_quality import (
        target_observation_mask,
    )

    target = torch.tensor([[1, 0], [0, 1]])
    reliability = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)

    assert target_observation_mask(target, reliability).tolist() == [
        [True, True],
        [False, True],
    ]


def test_binary_metrics_apply_one_literal_pixel_mask() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_target_quality import (
        binary_metrics,
    )

    probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float64)
    logits = torch.logit(probabilities)
    target = torch.tensor([1, 0, 1, 0])
    include = torch.tensor([True, False, True, True])

    metrics = binary_metrics(logits, target, include)

    assert metrics["pixel_count"] == 3
    assert metrics["positive_count"] == 2
    assert metrics["avg_precision"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.22)
    assert metrics["loss"] == pytest.approx(0.344049, abs=1e-6)


def test_risk_coverage_retains_highest_quality_events_first() -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_target_quality import (
        risk_coverage,
    )

    records = [
        {"quality": 0.2, "error": 10.0},
        {"quality": 0.9, "error": 1.0},
        {"quality": 0.1, "error": 20.0},
        {"quality": 0.8, "error": 3.0},
    ]

    assert risk_coverage(records, quality_key="quality", error_key="error") == [
        {"coverage": 0.25, "sample_count": 1, "mean_error": 1.0},
        {"coverage": 0.5, "sample_count": 2, "mean_error": 2.0},
        {"coverage": 0.75, "sample_count": 3, "mean_error": 14 / 3},
        {"coverage": 1.0, "sample_count": 4, "mean_error": 8.5},
    ]


def test_target_quality_dataset_aligns_target_day_and_center_crop(tmp_path) -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_target_quality import (
        TargetQualityDataset,
    )
    from reproductions.wsts_fast_track.viirs_reliability import (
        FIXED_2021_MODEL_GATE,
        FIXED_2021_TARGET_GATE,
    )

    input_records = []
    target_records = []
    for position, ((event, model_day), (_, target_day)) in enumerate(
        zip(FIXED_2021_MODEL_GATE, FIXED_2021_TARGET_GATE, strict=True)
    ):
        relative = f"{event}/{target_day}.npz"
        target_records.append(
            {
                "event": event,
                "day": target_day,
                "output": relative,
                "reliable_fraction": 0.5,
                "mean_reliable_age_hours": 4.0,
            }
        )
        input_records.append(
            {
                "event": event,
                "day": model_day,
                "reliable_fraction": 0.75,
                "mean_reliable_age_hours": 3.0,
            }
        )
        if position == 0:
            path = tmp_path / relative
            path.parent.mkdir(parents=True)
            full = np.zeros((4, 4), dtype=np.uint8)
            full[1:3, 1:3] = np.array([[1, 0], [0, 1]], dtype=np.uint8)
            np.savez(path, reliability=full, age_hours=np.full((4, 4), 4.0))
            target_records[-1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "target",
                "samples": target_records,
            }
        ),
        encoding="utf-8",
    )

    class InputDataset:
        samples = tuple(
            (event, day, position, input_records[position])
            for position, (event, day) in enumerate(FIXED_2021_MODEL_GATE)
        )

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            return torch.zeros((1, 41, 2, 2)), torch.tensor([[1, 0], [0, 0]])

    dataset = TargetQualityDataset(InputDataset(), tmp_path)
    packed, target, reliability, sample_index = dataset[0]

    assert len(dataset) == 24
    assert packed.shape == (1, 41, 2, 2)
    assert target.tolist() == [[1, 0], [0, 0]]
    assert reliability.tolist() == [[1, 0], [0, 1]]
    assert sample_index == 0
    # A fresh loader must verify the manifest before decoding altered NPZ bytes.
    (tmp_path / target_records[0]["output"]).write_bytes(b'corrupted NPZ')
    fresh = TargetQualityDataset(InputDataset(), tmp_path)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        fresh[0]


def test_natural_reliability_rejects_corrupt_map_before_numpy_load(tmp_path) -> None:
    from reproductions.wsts_fast_track.evaluate_viirs_reliability import NaturalReliabilityDataset

    path = tmp_path / 'sample.npz'
    path.write_bytes(b'corrupted bytes')
    record = dict(output=path.name, sha256=hashlib.sha256(b'original bytes').hexdigest())
    dataset = object.__new__(NaturalReliabilityDataset)
    dataset.root = tmp_path
    dataset.samples = (('event', '2021-01-01', 0, record),)
    dataset.standard = [(torch.zeros((1, 41, 2, 2)), torch.zeros((2, 2)))]
    dataset._reliability_cache = {}
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        dataset[0]
