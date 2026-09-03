from __future__ import annotations

from types import SimpleNamespace

import pytest

from reproductions.wsts_fast_track import corrected_baselines
from reproductions.wsts_fast_track.environment_dro import resolve_dataset_index


def test_corrected_baselines_are_from_scratch_single_variable_controls() -> None:
    assert corrected_baselines.corrected_baseline_spec("B0").training_policy == "clean"
    assert corrected_baselines.corrected_baseline_spec("B1").experiment_id == "C02"
    assert corrected_baselines.corrected_baseline_spec("B2").training_policy == "fire"
    assert (
        corrected_baselines.corrected_baseline_spec("B3").training_policy
        == "fire-block"
    )
    assert (
        corrected_baselines.corrected_baseline_spec("B4").training_policy
        == "year-balanced-fire-block"
    )
    assert all(
        spec.seed == 0 and spec.max_steps == 3_000
        for spec in corrected_baselines.CORRECTED_BASELINES.values()
    )

    with pytest.raises(ValueError, match="unknown corrected baseline"):
        corrected_baselines.corrected_baseline_spec("P00")


def test_install_corrected_baseline_sets_resolver_before_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        find_image_index_from_dataset_index = object()

    events: list[tuple[str, object]] = []

    def fake_import(name: str) -> object:
        assert name == "dataloader.FireSpreadDataset"
        return SimpleNamespace(FireSpreadDataset=FakeDataset)

    def record_fire(_root: object) -> None:
        events.append(("fire", FakeDataset.find_image_index_from_dataset_index))

    def record_fire_block(_root: object) -> None:
        events.append(("fire-block", FakeDataset.find_image_index_from_dataset_index))

    monkeypatch.setattr(corrected_baselines.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        corrected_baselines, "install_training_fire_dropout", record_fire
    )
    monkeypatch.setattr(
        corrected_baselines,
        "install_training_fire_and_block_dropout",
        record_fire_block,
    )

    corrected_baselines.install_corrected_baseline("/fake/upstream", "B2")
    assert events == [("fire", resolve_dataset_index)]

    corrected_baselines.install_corrected_baseline("/fake/upstream", "B3")
    assert events[-1] == ("fire-block", resolve_dataset_index)


def test_clean_corrected_baseline_installs_no_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        find_image_index_from_dataset_index = object()

    monkeypatch.setattr(
        corrected_baselines.importlib,
        "import_module",
        lambda _name: SimpleNamespace(FireSpreadDataset=FakeDataset),
    )
    monkeypatch.setattr(
        corrected_baselines,
        "install_training_fire_dropout",
        lambda _root: pytest.fail("clean baseline installed FireDrop"),
    )
    monkeypatch.setattr(
        corrected_baselines,
        "install_training_fire_and_block_dropout",
        lambda _root: pytest.fail("clean baseline installed BlockDrop"),
    )

    corrected_baselines.install_corrected_baseline("/fake/upstream", "B0")

    assert FakeDataset.find_image_index_from_dataset_index is resolve_dataset_index


def test_year_balanced_baseline_changes_only_corruption_and_sampler_installers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        find_image_index_from_dataset_index = object()

    events: list[str] = []
    monkeypatch.setattr(
        corrected_baselines.importlib,
        "import_module",
        lambda _name: SimpleNamespace(FireSpreadDataset=FakeDataset),
    )
    monkeypatch.setattr(
        corrected_baselines,
        "install_training_fire_and_block_dropout",
        lambda _root: events.append("fire-block"),
    )
    monkeypatch.setattr(
        corrected_baselines,
        "install_balanced_year_training_loader",
        lambda _root: events.append("year-balanced"),
    )

    corrected_baselines.install_corrected_baseline("/fake/upstream", "B4")

    assert events == ["fire-block", "year-balanced"]
