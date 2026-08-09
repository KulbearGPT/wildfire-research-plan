import json
import subprocess
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

from control import (  # noqa: E402
    validate_environment,
    verify_inventory,
    verify_upstream,
    validate_overrides,
    write_json_atomic,
)


FROZEN_COUNTS = {2018: 176, 2019: 74, 2020: 201, 2021: 156}
LOCK_PATH = SCRIPTS_DIR.parent / "upstream.lock.json"
SCIENTIFIC_OVERRIDES = {
    "data.data_fold_id": 2,
    "data.features_to_keep": None,
    "data.n_leading_observations": 1,
    "data.remove_duplicate_features": True,
    "trainer.max_steps": 500,
    "do_test": False,
}


def _environment(artifacts_root: Path, *, num_workers: int = 64) -> dict[str, object]:
    return {
        "data.data_dir": r"D:\WildFire Project\data\hdf5",
        "trainer.default_root_dir": str((artifacts_root / "run-001").resolve()),
        "data.num_workers": num_workers,
        "WANDB_MODE": "disabled",
        "timing_logging": True,
        "progress_logging": True,
    }


def _write_inventory(root: Path, counts: dict[int, int] = FROZEN_COUNTS) -> None:
    for year, count in counts.items():
        year_dir = root / str(year)
        year_dir.mkdir(parents=True)
        for index in range(count):
            (year_dir / f"event_{index}.hdf5").touch()


def test_verify_inventory_accepts_exact_frozen_direct_file_counts(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    (tmp_path / "2018" / "nested").mkdir()
    (tmp_path / "2018" / "nested" / "ignored.hdf5").touch()
    (tmp_path / "2018" / "note.txt").touch()

    assert verify_inventory(tmp_path) == {
        "2018": 176,
        "2019": 74,
        "2020": 201,
        "2021": 156,
        "total": 607,
    }


@pytest.mark.parametrize(
    ("counts", "extra_year", "message"),
    [
        ({2018: 175, 2019: 74, 2020: 201, 2021: 156}, None, "2018"),
        ({2018: 177, 2019: 74, 2020: 201, 2021: 156}, None, "2018"),
        (FROZEN_COUNTS, 2022, "2022"),
    ],
)
def test_verify_inventory_default_rejects_missing_extra_or_wrong_year_hdf5(
    tmp_path: Path,
    counts: dict[int, int],
    extra_year: int | None,
    message: str,
) -> None:
    _write_inventory(tmp_path, counts)
    if extra_year is not None:
        wrong_year = tmp_path / str(extra_year)
        wrong_year.mkdir()
        (wrong_year / "event.hdf5").touch()

    with pytest.raises(ValueError, match=message):
        verify_inventory(tmp_path)


def test_validate_overrides_accepts_only_the_exact_allowlisted_mapping() -> None:
    validate_overrides(SCIENTIFIC_OVERRIDES)


@pytest.mark.parametrize(
    "change",
    [
        {"do_test": True},
        {"data.data_fold_id": 1},
        {"data.features_to_keep": "weather"},
        {"trainer.max_steps": 501},
        {"data.batch_size": 32},
        {"trainer.precision": "16-mixed"},
        {"optimizer": "sgd"},
        {"model": "unet_resnet50"},
    ],
)
def test_validate_overrides_rejects_scientific_or_type_changes(
    change: dict[str, object],
) -> None:
    overrides = dict(SCIENTIFIC_OVERRIDES)
    overrides.update(change)

    with pytest.raises(ValueError):
        validate_overrides(overrides)


def test_validate_overrides_rejects_equivalent_value_with_wrong_type() -> None:
    overrides = dict(SCIENTIFIC_OVERRIDES)
    overrides["trainer.max_steps"] = True

    with pytest.raises(ValueError, match="type"):
        validate_overrides(overrides)


def test_write_json_atomic_replaces_existing_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.json"
    target.parent.mkdir()
    target.write_text('{"stale": true}', encoding="utf-8")

    write_json_atomic(target, {"fresh": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8")) == {"fresh": [1, 2, 3]}
    assert list(target.parent.glob(".result.json.*.tmp")) == []


def test_verify_upstream_accepts_only_a_clean_pinned_checkout_with_injected_runner(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = "expected-pin\n" if command[-1] == "HEAD" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    assert verify_upstream(tmp_path, "expected-pin", command_runner=run_command) == "expected-pin"
    assert calls == [
        ["git", "-C", str(tmp_path.resolve()), "rev-parse", "HEAD"],
        ["git", "-C", str(tmp_path.resolve()), "status", "--porcelain"],
    ]


def test_verify_upstream_rejects_a_dirty_checkout_with_injected_runner(tmp_path: Path) -> None:
    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        stdout = "expected-pin\n" if command[-1] == "HEAD" else " M src/train.py\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with pytest.raises(ValueError, match="upstream checkout is dirty"):
        verify_upstream(tmp_path, "expected-pin", command_runner=run_command)


@pytest.mark.parametrize("failed_command", ["HEAD", "--porcelain"])
def test_verify_upstream_rejects_injected_git_command_errors(
    tmp_path: Path, failed_command: str
) -> None:
    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[-1] == failed_command:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="git failed")
        return subprocess.CompletedProcess(command, 0, stdout="expected-pin\n", stderr="")

    with pytest.raises(ValueError, match="cannot determine upstream|cannot verify upstream"):
        verify_upstream(tmp_path, "expected-pin", command_runner=run_command)


def test_verify_upstream_rejects_a_commit_mismatch_without_a_real_repository(
    tmp_path: Path,
) -> None:
    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="not-the-pin\n", stderr="")

    with pytest.raises(ValueError, match="upstream commit mismatch"):
        verify_upstream(tmp_path, "expected-pin", command_runner=run_command)


def test_validate_environment_accepts_only_local_execution_controls(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "reproductions"

    for workers in (64, 8, 4, 0):
        validate_environment(_environment(artifacts_root, num_workers=workers), artifacts_root)


@pytest.mark.parametrize(
    "change",
    [
        {"data.data_dir": r"D:\other\data"},
        {"trainer.default_root_dir": "relative-output"},
        {"data.num_workers": 2},
        {"WANDB_MODE": "online"},
        {"timing_logging": False},
        {"progress_logging": False},
    ],
)
def test_validate_environment_rejects_unapproved_local_execution_changes(
    tmp_path: Path, change: dict[str, object]
) -> None:
    artifacts_root = tmp_path / "artifacts" / "reproductions"
    environment = _environment(artifacts_root)
    environment.update(change)

    with pytest.raises(ValueError):
        validate_environment(environment, artifacts_root)


def test_validate_overrides_cli_prints_compact_json_and_reports_validation_errors() -> None:
    command = [sys.executable, str(SCRIPTS_DIR / "control.py"), "validate-overrides"]
    valid = subprocess.run(
        [*command, json.dumps(SCIENTIFIC_OVERRIDES)],
        text=True,
        capture_output=True,
        check=False,
    )
    invalid = subprocess.run(
        [*command, json.dumps({**SCIENTIFIC_OVERRIDES, "do_test": True})],
        text=True,
        capture_output=True,
        check=False,
    )

    assert valid.returncode == 0
    assert valid.stdout == '{"valid":true}\n'
    assert valid.stderr == ""
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert "do_test" in invalid.stderr


def _make_git_repository(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "fixture.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.email=tests@example.invalid",
            "-c", "user.name=Test Fixture", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def test_inventory_and_upstream_cli_report_success_and_validation_errors(tmp_path: Path) -> None:
    _write_inventory(tmp_path / "data")
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = _make_git_repository(repository)
    command = [sys.executable, str(SCRIPTS_DIR / "control.py")]

    inventory = subprocess.run(
        [*command, "inventory", str(tmp_path / "data")], text=True, capture_output=True, check=False
    )
    upstream = subprocess.run(
        [*command, "upstream", str(repository), commit], text=True, capture_output=True, check=False
    )
    bad_inventory = subprocess.run(
        [*command, "inventory", str(tmp_path / "missing")], text=True, capture_output=True, check=False
    )
    bad_upstream = subprocess.run(
        [*command, "upstream", str(repository), "not-the-pin"], text=True, capture_output=True, check=False
    )

    assert inventory.returncode == 0
    assert json.loads(inventory.stdout) == {"2018": 176, "2019": 74, "2020": 201, "2021": 156, "total": 607}
    assert upstream.returncode == 0
    assert json.loads(upstream.stdout) == {"commit": commit}
    assert bad_inventory.returncode == 2
    assert "data root" in bad_inventory.stderr
    assert bad_upstream.returncode == 2
    assert "upstream commit mismatch" in bad_upstream.stderr


def test_upstream_lock_exposes_all_frozen_provenance_fields() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock["code"] == {
        "url": "https://github.com/slahrichi/WildfireSpreadTS.git",
        "commit": "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad",
    }
    assert lock["weights"] == {
        "url": "https://huggingface.co/saadlahrichi/WSTSPlus",
        "revision": "acf70a37394849f4ec8d108a51d6f4325a554d0a",
    }
    assert lock["paper"]["target"] == "0.460 +/- 0.084"
    assert lock["calibration"] == {
        "fold": 2,
        "max_steps": 500,
        "test_loader_allowed": False,
        "purpose": "timing only",
    }
