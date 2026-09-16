"""Targeted runtime installation checks; execute in the Slurm test allocation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reproductions.wsts_fast_track import runtime


@pytest.mark.parametrize("experiment_id", ["C00", "C02"])
def test_temporal_class_export_is_installed_only_for_c02(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, experiment_id: str
) -> None:
    class FireSpreadDataModule:
        pass

    class SMPTempModel:
        pass

    models = SimpleNamespace()
    modules = {
        "dataloader.FireSpreadDataModule": SimpleNamespace(
            FireSpreadDataModule=FireSpreadDataModule
        ),
        "dataloader.FireSpreadDataset": SimpleNamespace(),
        "dataloader.utils": SimpleNamespace(),
        "models": models,
        "models.SMPTempModel": SimpleNamespace(SMPTempModel=SMPTempModel),
    }
    imported = []

    def import_module(name):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime.sys, "path", list(runtime.sys.path))
    runtime._install_runtime_contract(
        tmp_path, experiment_id, (np.zeros(23), np.ones(23), np.zeros(23))
    )

    if experiment_id == "C02":
        # Lightning's models.SMPTempModel path must resolve to the class,
        # even when the patched upstream package does not re-export it.
        assert models.SMPTempModel is SMPTempModel
        assert imported[-2:] == ["models", "models.SMPTempModel"]
    else:
        assert not hasattr(models, "SMPTempModel")
        assert "models.SMPTempModel" not in imported
    assert FireSpreadDataModule.split_fires(0, True) == (
        list(runtime.TRAIN_YEARS), [2021], [2021]
    )
