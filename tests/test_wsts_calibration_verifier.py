import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "reproductions"
    / "wsts_res18_unet_t1"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_calibration import (  # noqa: E402
    build_exact_expected_command,
    derive_dynamic_positive_weight_provenance,
    independent_epoch_boundaries,
    independent_epoch_projection,
    independent_effective_config,
    independent_progress,
    independent_timing,
    validate_command_lineage,
    validate_exact_command,
    validate_import_scope_contents,
    validate_patch_text,
)


def _event(seconds: float, text: str) -> str:
    return json.dumps({"seconds": seconds, "stream": "stdout", "text": text})


def test_independent_progress_handles_lightning_completed_epoch_reset() -> None:
    events = "\n".join(
        [
            _event(1.0, "Epoch 0: 0%| | 0/2"),
            _event(2.0, "Epoch 0: 50%| | 1/2"),
            _event(3.0, "Epoch 0: 100%| | 2/2"),
            _event(4.0, "Epoch 0: 0%| | 0/2"),
            _event(5.0, "Epoch 1: 0%| | 0/2"),
            _event(6.0, "Epoch 1: 50%| | 1/2"),
        ]
    )

    assert independent_progress(events) == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 6.0)]


def test_independent_timing_excludes_warmup_through_step49() -> None:
    result = independent_timing(
        [(0, 0.0), (49, 4.9), (50, 5.2), (51, 5.6), (52, 6.0)],
        wall_seconds=10.0,
        validation_seconds=1.0,
    )

    assert result["median_step_seconds"] == pytest.approx(0.4)
    assert result["instantaneous_samples_per_second"] == pytest.approx(160.0)
    assert result["compute_only_10000_seconds"] == pytest.approx(4000.0)


def test_independent_epoch_projection_recomputes_boundaries_cycles_and_wall_model() -> None:
    events = "\n".join(
        [
            _event(1.0, "Epoch 0: 0%| | 0/2"),
            _event(2.0, "Epoch 0: 50%| | 1/2"),
            _event(3.0, "Epoch 0: 100%| | 2/2"),
            _event(4.0, "Epoch 0: 0%| | 0/2"),
            _event(5.0, "Epoch 1: 0%| | 0/2"),
            _event(6.0, "Epoch 1: 50%| | 1/2"),
            _event(7.0, "Epoch 1: 100%| | 2/2"),
            _event(8.0, "Epoch 1: 0%| | 0/2"),
            _event(9.0, "Epoch 2: 0%| | 0/2"),
            _event(10.0, "Epoch 2: 50%| | 1/2"),
        ]
    )
    boundaries = independent_epoch_boundaries(events)
    result = independent_epoch_projection(
        boundaries,
        startup_seconds=1.0,
        median_step_seconds=0.1,
        p25_step_seconds=0.08,
        p75_step_seconds=0.12,
        wall_seconds=10.0,
        observed_steps=5,
    )

    assert boundaries == [(0, 1, 2.0, 2), (1, 3, 6.0, 2), (2, 5, 10.0, 2)]
    assert result["epoch_cycle_seconds"] == [4.0, 4.0]
    assert result["epoch_aware_10000_central_seconds"] == pytest.approx(20_001.0)
    assert result["naive_wall_linear_10000_seconds"] == pytest.approx(20_000.0)


def test_independent_effective_config_reads_dynamic_positive_weight(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  init_args:\n    pos_class_weight: 608.4653828020165\n",
        encoding="utf-8",
    )

    assert independent_effective_config(config) == 608.4653828020165


def test_exact_expected_command_rejects_any_value_order_or_extra_argument(
    tmp_path: Path,
) -> None:
    run = (tmp_path / "run").resolve()
    derived = (tmp_path / "WildfireSpreadTS-res18-runtime").resolve()
    expected = build_exact_expected_command(run, derived)
    validate_exact_command(expected, expected)

    mutations = [
        ["wrong-python", *expected[1:]],
        [expected[0], "wrong-entrypoint", *expected[2:]],
        [*expected[:3], "wrong-derived-root", *expected[4:]],
        [*expected[:4], "--config=wrong-config", *expected[5:]],
        [*expected[:7], "--data.data_dir=wrong-data", *expected[8:]],
        [*expected[:8], "--data.data_fold_id=1", *expected[9:]],
        [*expected[:14], "--trainer.default_root_dir=wrong-run", expected[15]],
        [*expected, "--optimizer.init_args.lr=0.01"],
        [*expected[:8], expected[9], expected[8], *expected[10:]],
    ]
    for mutation in mutations:
        with pytest.raises(ValueError, match="exact expected command"):
            validate_exact_command(mutation, expected)


def test_command_lineage_cross_checks_all_markers_and_hashes(tmp_path: Path) -> None:
    run = (tmp_path / "run").resolve()
    run.mkdir()
    derived = (tmp_path / "WildfireSpreadTS-res18-runtime").resolve()
    command = build_exact_expected_command(run, derived)
    command_hash = __import__("hashlib").sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {"command": command, "command_sha256": command_hash}
    for name in (
        "started.json",
        "launch.lock.json",
        "worker-recovery-authorization.json",
        "effective-command.json",
    ):
        (run / name).write_text(json.dumps(payload), encoding="utf-8")
    (run / "timing.json").write_text(
        json.dumps({"command_sha256": command_hash}), encoding="utf-8"
    )
    global_lock = tmp_path / "fold2-calibration-worker-recovery.lock.json"
    global_lock.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_command_lineage(run, global_lock, command) == command_hash
    (run / "effective-command.json").write_text(
        json.dumps({**payload, "command_sha256": "tampered"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="command lineage"):
        validate_command_lineage(run, global_lock, command)


def _official_init_text() -> str:
    return "\n".join(
        [
            "from .BaseModel import BaseModel",
            "from .ConvLSTMLightning import ConvLSTMLightning",
            "from .LogisticRegression import LogisticRegression",
            "from .SMPModel import SMPModel",
            "from .UTAELightning import UTAELightning",
            "from .SwinUnetLightning import SwinUnetLightning",
            "from .SwinUnetTempLightning import SwinUnetTempLightning",
            "from .UTAELightningDumb import UTAELightningDumb",
            "from .TransUnetLightning import TransUnetLightning",
            "from .SMPTempModel import SMPTempModel ",
            "from .SegFormerLightning import SegFormerLightning",
        ]
    )


def test_import_scope_provenance_rejects_tampered_derived_body_or_patch() -> None:
    original = _official_init_text()
    derived = "".join(original.splitlines(keepends=True)[:4])
    validate_import_scope_contents(original, derived)
    with pytest.raises(ValueError, match="derived initializer"):
        validate_import_scope_contents(original, derived + "\nSCIENTIFIC_CHANGE = True")

    patch_path = SCRIPTS_DIR.parent / "patches" / "res18_import_scope.patch"
    patch = patch_path.read_text(encoding="utf-8")
    validate_patch_text(patch)
    with pytest.raises(ValueError, match="patch contract"):
        validate_patch_text(patch.replace("SwinUnetLightning", "SMPModel"))


def test_dynamic_positive_weight_provenance_is_derived_from_yaml_ast_and_config() -> None:
    source_yaml = "model:\n  init_args:\n    pos_class_weight: 236\n"
    effective = "model:\n  init_args:\n    pos_class_weight: 608.4653828020165\n"
    train = (
        "class CLI:\n"
        "    def before_instantiate_classes(self):\n"
        "        fire_rate = 1 - missing_values_rates[-1]\n"
        "        pos_class_weight = float(1 / fire_rate)\n"
        "        self.config.model.init_args.pos_class_weight = pos_class_weight\n"
    )
    result = derive_dynamic_positive_weight_provenance(source_yaml, train, effective)
    assert result == {
        "source_yaml_pos_class_weight": 236.0,
        "effective_pos_class_weight": 608.4653828020165,
        "official_dynamic_override": True,
        "official_override_expression": "float(1 / fire_rate)",
    }
    with pytest.raises(ValueError, match="source YAML"):
        derive_dynamic_positive_weight_provenance(
            source_yaml.replace("236", "237"), train, effective
        )
    with pytest.raises(ValueError, match="before_instantiate_classes"):
        derive_dynamic_positive_weight_provenance(
            source_yaml, train.replace("1 / fire_rate", "1 + fire_rate"), effective
        )
