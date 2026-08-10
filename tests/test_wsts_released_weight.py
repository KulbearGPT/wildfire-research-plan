import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_released_weight import (  # noqa: E402
    _strict_load_preflight,
    build_weight_command,
    summarize_filename_aps,
    validate_download,
    validate_weight_command,
    validate_weight_manifest,
    validate_weight_output,
)
import evaluate_released_weight as weight_controller  # noqa: E402
from verify_released_weight import validate_weight_result  # noqa: E402
import verify_released_weight as weight_verifier  # noqa: E402


FILENAMES = [
    "fold0_testAP0.528.pth",
    "fold1_testAP0.426.pth",
    "fold2_testAP0.571.pth",
    "fold3_testAP0.307.pth",
    "fold4_testAP0.483.pth",
    "fold5_testAP0.322.pth",
    "fold6_testAP0.577.pth",
    "fold7_testAP0.474.pth",
    "fold8_testAP0.478.pth",
    "fold9_testAP0.471.pth",
    "fold10_testAP0.324.pth",
    "fold11_testAP0.474.pth",
]


def test_strict_load_preflight_propagates_disabled_wandb_environment(
    tmp_path: Path,
) -> None:
    child = tmp_path / "strict_child.py"
    child.write_text(
        "import os\n"
        "if os.environ.get('WANDB_MODE') != 'disabled':\n"
        "    raise SystemExit(7)\n"
        "print('WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=1')\n",
        encoding="utf-8",
    )

    result = _strict_load_preflight(
        [sys.executable, str(child), "--upstream-root", "unused", "--weights-path", "unused"]
    )

    assert result["status"] == "pass"
    assert result["test_loader_invoked"] is False


def _manifest() -> list[dict[str, object]]:
    items = []
    for name in FILENAMES:
        item = {
            "type": "file",
            "path": f"trained_model_weights/Res18Unet_T1/All/{name}",
            "size": 57_889_221,
            "lfs": {"oid": "0" * 64},
        }
        if name == "fold2_testAP0.571.pth":
            item["lfs"] = {
                "oid": "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
            }
        items.append(item)
    return items


def test_manifest_selects_exact_pinned_fold2_weight_and_aggregate() -> None:
    selected = validate_weight_manifest(_manifest())

    assert selected["path"] == (
        "trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth"
    )
    assert selected["size"] == 57_889_221
    assert selected["lfs_sha256"] == (
        "e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec"
    )
    aggregate = summarize_filename_aps(FILENAMES)
    assert aggregate["count"] == 12
    assert aggregate["mean"] == pytest.approx(0.4529166666666667)
    assert aggregate["population_std"] == pytest.approx(0.0882717946017992)


def test_manifest_rejects_missing_duplicate_or_wrong_fold2_hash() -> None:
    with pytest.raises(ValueError, match="exactly twelve"):
        validate_weight_manifest(_manifest()[:-1])
    duplicate = _manifest()
    duplicate[-1]["path"] = duplicate[0]["path"]
    with pytest.raises(ValueError, match="unique folds"):
        validate_weight_manifest(duplicate)
    wrong_hash = _manifest()
    wrong_hash[2]["lfs"] = {"oid": "f" * 64}
    with pytest.raises(ValueError, match="Fold 2 weight"):
        validate_weight_manifest(wrong_hash)


def test_download_validation_checks_exact_size_and_sha(tmp_path: Path) -> None:
    weight = tmp_path / "fold2.pth"
    weight.write_bytes(b"official-weight")
    digest = hashlib.sha256(b"official-weight").hexdigest()

    validate_download(weight, expected_size=15, expected_sha256=digest)
    with pytest.raises(ValueError, match="size"):
        validate_download(weight, expected_size=16, expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_download(weight, expected_size=15, expected_sha256="0" * 64)


def test_weight_command_is_test_only_on_fold2_all_t1(tmp_path: Path) -> None:
    command = build_weight_command(
        run_directory=(tmp_path / "run").resolve(),
        weight_path=(tmp_path / "fold2.pth").resolve(),
        python_executable=(tmp_path / "env" / "python.exe").resolve(),
        entrypoint=(SCRIPTS_DIR / "official_weight_entrypoint.py").resolve(),
        upstream_root=(tmp_path / "runtime").resolve(),
        data_root=Path(r"D:\WildFire Project\data\hdf5"),
    )

    assert "--data.data_fold_id=2" in command
    assert "--data.features_to_keep=null" in command
    assert "--data.n_leading_observations=1" in command
    assert "--data.num_workers=8" in command
    assert "--do_train=false" in command
    assert "--do_test=false" in command
    assert not any("max_steps" in item or "do_predict" in item or "do_validate" in item for item in command)
    validate_weight_command(command, (tmp_path / "run").resolve(), (tmp_path / "fold2.pth").resolve())

    changed = ["--do_train=true" if item == "--do_train=false" else item for item in command]
    with pytest.raises(ValueError, match="exact released-weight command"):
        validate_weight_command(
            changed, (tmp_path / "run").resolve(), (tmp_path / "fold2.pth").resolve()
        )


def test_weight_output_requires_strict_load_and_test_without_fit_or_validation() -> None:
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "│ test_f1 │ 0.4 │\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
    )

    result = validate_weight_output(output, 0)

    assert result["strict_load"] is True
    assert result["test_metrics"]["test_AP"] == pytest.approx(0.5708)
    assert result["peak_allocated_bytes"] == 1_700_000_000


@pytest.mark.parametrize(
    ("extra", "exit_code", "message"),
    [
        ("", 1, "exit code"),
        ("Epoch 0: 1/121", 0, "training"),
        ("Validation DataLoader 0: 1/22", 0, "validation"),
        ("Predicting DataLoader 0: 1/1", 0, "predict"),
    ],
)
def test_weight_output_fails_closed_on_non_test_actions(
    extra: str, exit_code: int, message: str
) -> None:
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
        + extra
    )
    with pytest.raises(ValueError, match=message):
        validate_weight_output(output, exit_code)


def test_released_weight_result_requires_strict_load_test_only_and_ap_delta() -> None:
    result = validate_weight_result(
        {
            "exit_code": 0,
            "strict_load": True,
            "train_invoked": False,
            "validation_invoked": False,
            "predict_invoked": False,
            "test_metrics": {"test_AP": 0.5708, "test_f1": 0.4},
            "filename_ap": 0.571,
        }
    )

    assert result["test_AP"] == pytest.approx(0.5708)
    assert result["filename_ap_absolute_difference"] == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="strict"):
        validate_weight_result(
            {
                "exit_code": 0,
                "strict_load": False,
                "train_invoked": False,
                "validation_invoked": False,
                "predict_invoked": False,
                "test_metrics": {"test_AP": 0.5708},
                "filename_ap": 0.571,
            }
        )


def test_weight_launch_dependency_requires_completed_and_independently_verified_full_run(
    tmp_path: Path,
) -> None:
    validate = getattr(weight_controller, "validate_full_run_dependency", None)
    assert validate is not None, "full-run dependency validator is required"
    run = (tmp_path / "fold2-full-run").resolve()
    run.mkdir()
    lock = tmp_path / "fold2-full-launch.lock.json"
    lock.write_text(json.dumps({"run_directory": str(run)}), encoding="utf-8")
    (run / "completed.json").write_text(
        json.dumps({"status": "pass", "exit_code": 0}), encoding="utf-8"
    )
    (run / "full-result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "optimizer_steps": 10_000,
                "command_sha256": "full-command",
                "checkpoint_sha256": "best-checkpoint",
            }
        ),
        encoding="utf-8",
    )
    independent = run / "independent-verification.json"
    independent.write_text(
        json.dumps(
            {
                "status": "pass",
                "optimizer_steps": 10_000,
                "command_sha256": "full-command",
                "checkpoint_sha256": "best-checkpoint",
            }
        ),
        encoding="utf-8",
    )

    result = validate(lock)
    assert result["independent_status"] == "pass"
    independent.unlink()
    with pytest.raises(ValueError) as error:
        validate(lock)
    assert str(error.value) == "full Fold-2 independent verification is missing"


def test_independent_weight_verifier_builds_exact_command_and_requires_all_lineage(
    tmp_path: Path,
) -> None:
    builder = getattr(weight_verifier, "build_expected_weight_command", None)
    validate = getattr(weight_verifier, "verify_weight_command_lineage", None)
    assert builder is not None, "independent weight command builder is required"
    assert validate is not None, "independent weight lineage validator is required"
    run = (tmp_path / "weight-run").resolve()
    derived = (tmp_path / "WildfireSpreadTS-res18-runtime").resolve()
    weight = (tmp_path / "fold2_testAP0.571.pth").resolve()
    command = builder(run, derived, weight)
    command_hash = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {"command": command, "command_sha256": command_hash}
    payloads = {
        "launch": dict(payload),
        "run_lock": dict(payload),
        "preflight": dict(payload),
        "started": dict(payload),
        "effective": dict(payload),
        "result": {"command_sha256": command_hash},
    }

    validate(payloads, command, command_hash)
    del payloads["run_lock"]
    with pytest.raises(ValueError) as error:
        validate(payloads, command, command_hash)
    assert str(error.value) == "run_lock marker is missing"


def test_independent_weight_verifier_reconstructs_raw_strict_test_evidence() -> None:
    reconstruct = getattr(weight_verifier, "reconstruct_weight_evidence", None)
    assert reconstruct is not None, "independent raw weight evidence parser is required"
    output = (
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=123\n"
        "Testing DataLoader 0: 100%|##########| 156/156\n"
        "│ test_AP │ 0.5708 │\n"
        "│ test_f1 │ 0.4 │\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1700000000\n"
    )

    assert reconstruct(output, 0) == {
        "exit_code": 0,
        "strict_load": True,
        "loaded_tensor_count": 123,
        "train_invoked": False,
        "validation_invoked": False,
        "predict_invoked": False,
        "test_metrics": {"test_AP": pytest.approx(0.5708), "test_f1": pytest.approx(0.4)},
        "peak_allocated_bytes": 1_700_000_000,
    }
