import sys
import csv
import errno
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import released_weight_contract as contract  # noqa: E402
import verify_released_weight as fold_verifier  # noqa: E402
import verify_weight_campaign as verifier  # noqa: E402


@pytest.fixture(scope="module")
def specs() -> tuple[contract.WeightSpec, ...]:
    return contract.load_pinned_manifest(
        REPOSITORY_ROOT
        / "reproductions"
        / "wsts_res18_unet_t1"
        / "official_weights_manifest.json"
    )


def _raw_output(metrics: dict[str, float]) -> str:
    lines = [
        "WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1 tensors=182",
        "Testing DataLoader 0: 100%",
        *(f"{name}  {value!r}" for name, value in metrics.items()),
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1000",
    ]
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(path: Path) -> dict[str, object]:
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _build_synthetic_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> tuple[Path, list[dict[str, object]]]:
    campaign_root = tmp_path / "campaigns"
    campaign_directory = campaign_root / "official-weight-12fold-synthetic"
    run_root = tmp_path / "fold-runs"
    campaign_directory.mkdir(parents=True)
    run_root.mkdir()
    expected: list[dict[str, object]] = []
    for spec in specs:
        ap = 0.10 + 0.05 * spec.fold_id
        metrics = {
            "test_AP": ap,
            "test_f1": 0.20 + 0.01 * spec.fold_id,
            "test_iou": 0.15 + 0.01 * spec.fold_id,
            "test_loss": 0.05 + 0.01 * spec.fold_id,
            "test_precision": 0.30 + 0.01 * spec.fold_id,
            "test_recall": 0.25 + 0.01 * spec.fold_id,
        }
        run = (run_root / f"fold{spec.fold_id}-weight-synthetic").resolve()
        run.mkdir()
        (run / "stdout.log").write_text(_raw_output(metrics), encoding="utf-8")
        (run / "stderr.log").write_text("", encoding="utf-8")
        (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
        raw_manifest = {
            "schema_version": 1,
            "entry_count": 3,
            "entries": [
                {
                    "path": name,
                    "bytes": (run / name).stat().st_size,
                    "sha256": _sha256(run / name),
                }
                for name in ("stdout.log", "stderr.log", "exit-code.txt")
            ],
        }
        independent = run / "independent-verification.json"
        independent.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "fold_id": spec.fold_id,
                    "test_metrics": metrics,
                    "wall_seconds": float(10 + spec.fold_id),
                    "raw_evidence_manifest": raw_manifest,
                    "raw_evidence_unchanged_during_verification": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        state = {
            "status": "pass",
            "fold_id": spec.fold_id,
            "mode": "adopt" if spec.fold_id == 2 else "launch",
            "run_directory": str(run),
            "independent_verification_sha256": _sha256(independent),
            "test_metrics": {name: 0.999 for name in metrics},
        }
        (campaign_directory / f"fold{spec.fold_id}.json").write_text(
            json.dumps(state) + "\n", encoding="utf-8"
        )
        expected.append(
            {
                "fold_id": spec.fold_id,
                "metrics": metrics,
                "wall_seconds": float(10 + spec.fold_id),
                "run": run,
            }
        )

    monkeypatch.setattr(verifier, "CAMPAIGN_ARTIFACTS_ROOT", campaign_root)
    monkeypatch.setattr(verifier, "WEIGHT_ARTIFACTS_ROOT", run_root)
    monkeypatch.setattr(
        verifier,
        "ADOPTED_FOLD2_RUN",
        (run_root / "fold2-weight-synthetic").resolve(),
    )
    monkeypatch.setattr(
        verifier, "weight_path_for_spec", lambda spec: tmp_path / spec.filename
    )

    def raw_manifest(run: Path, _weight: Path) -> dict[str, object]:
        entries = [
            {"path": name, "bytes": (run / name).stat().st_size, "sha256": _sha256(run / name)}
            for name in ("stdout.log", "stderr.log", "exit-code.txt")
        ]
        return {"schema_version": 1, "entry_count": 3, "entries": entries}

    def verify_fold(run: Path, _weight: Path, spec: contract.WeightSpec) -> dict[str, object]:
        output = (run / "stdout.log").read_text(encoding="utf-8") + "\n" + (
            run / "stderr.log"
        ).read_text(encoding="utf-8")
        evidence = fold_verifier.reconstruct_weight_evidence(output, 0)
        return {
            "status": "pass",
            "fold_id": spec.fold_id,
            "test_metrics": evidence["test_metrics"],
            "wall_seconds": float(10 + spec.fold_id),
            "raw_evidence_manifest": raw_manifest(run, Path("unused")),
            "raw_evidence_unchanged_during_verification": True,
        }

    monkeypatch.setattr(verifier, "capture_fold_raw_manifest", raw_manifest)
    monkeypatch.setattr(verifier, "verify_fold_run", verify_fold)
    return campaign_directory, expected


def test_aggregate_uses_raw_metrics_population_std_and_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, expected = _build_synthetic_campaign(tmp_path, monkeypatch, specs)

    result = verifier.verify_campaign(campaign_directory)

    values = [float(row["metrics"]["test_AP"]) for row in expected]
    publication = verifier.load_committed_publication(campaign_directory)
    generation = campaign_directory / publication["generation"]
    summary = json.loads((generation / verifier.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert summary["fold_count"] == 12
    assert summary["ap_mean"] == statistics.fmean(values)
    assert summary["ap_population_std"] == statistics.pstdev(values)
    assert summary["ap_population_std"] != statistics.stdev(values)
    assert summary["ap_min_fold_id"] == 0
    assert summary["ap_max_fold_id"] == 11
    assert summary["runtime_total_seconds"] == sum(range(10, 22))
    assert summary["runtime_median_seconds"] == statistics.median(range(10, 22))
    assert summary["ap_mean_minus_filename_mean"] == (
        summary["ap_mean"] - summary["filename_ap_mean"]
    )
    assert summary["ap_mean_minus_paper_mean"] == summary["ap_mean"] - 0.460
    with (generation / verifier.RESULTS_FILENAME).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["fold_id"]) for row in rows] == list(range(12))
    assert [int(row["test_year"]) for row in rows] == [spec.test_year for spec in specs]
    assert float(rows[0]["test_AP"]) == values[0]
    assert float(rows[0]["test_AP"]) != 0.999
    assert float(rows[2]["test_AP_minus_filename_ap"]) == values[2] - specs[2].filename_ap
    assert not list(campaign_directory.glob("*.tmp"))
    assert publication["status"] == "pass"
    assert publication["generation"].startswith("generations/")
    assert not (campaign_directory / verifier.RESULTS_FILENAME).exists()
    assert not (campaign_directory / verifier.SUMMARY_FILENAME).exists()
    assert not (campaign_directory / verifier.INDEPENDENT_FILENAME).exists()


def test_campaign_verifier_requires_recovery_chain_for_recovered_fold0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    fold0_path = campaign_directory / "fold0.json"
    fold0 = json.loads(fold0_path.read_text(encoding="utf-8"))
    fold0.update(
        {
            "recovered_via_finalize_existing": True,
            "scientific_child_relaunched": False,
        }
    )
    fold0_path.write_text(json.dumps(fold0) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recovery chain"):
        verifier.verify_campaign(campaign_directory)


def _install_synthetic_recovery_chain(
    tmp_path: Path,
    campaign_directory: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    fold0_path = campaign_directory / "fold0.json"
    state = json.loads(fold0_path.read_text(encoding="utf-8"))
    state.update(
        {
            "recovered_via_finalize_existing": True,
            "scientific_child_relaunched": False,
        }
    )
    fold0_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    run = Path(state["run_directory"])
    paths = {
        "schedule": campaign_directory / "schedule.json",
        "root_lock": campaign_directory.parent / verifier.GLOBAL_LOCK_NAME,
        "fold0_lock": campaign_directory / "fold0.lock.json",
        "failure": campaign_directory / "failure.json",
        "authorization": campaign_directory / verifier.EXTERNAL_RECOVERY_AUTHORIZATION_NAME,
        "resume": campaign_directory / "resume.lock.json",
        "receipt": run / verifier.RECOVERY_RECEIPT_NAME,
        "provenance": run / verifier.RECOVERY_PROVENANCE_NAME,
        "target": run / verifier.RECOVERY_TARGET_NAME,
        "completed": campaign_directory / "completed.json",
        "resume_completed": campaign_directory / "resume-completed.json",
    }
    paths["schedule"].write_text('{"schedule":"exact"}\n', encoding="utf-8")
    root_lock = {
        "schema_version": 1,
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "schedule_sha256": _sha256(paths["schedule"]),
        "single_campaign_no_retry": True,
        "sequential": True,
    }
    paths["root_lock"].write_text(json.dumps(root_lock) + "\n", encoding="utf-8")
    paths["fold0_lock"].write_text(
        json.dumps({**root_lock, "fold_id": 0, "mode": "launch"}) + "\n",
        encoding="utf-8",
    )
    paths["failure"].write_text('{"status":"fail","fold_id":0}\n', encoding="utf-8")
    source = tmp_path / "derived" / "test_pr_curve_data.npz"
    source.parent.mkdir()
    paths["target"].write_bytes(b"sealed-fold0-output")
    source_identity = {
        "path": str(source.absolute()),
        "bytes": paths["target"].stat().st_size,
        "sha256": _sha256(paths["target"]),
        "ctime_ns": 100,
        "mtime_ns": 200,
    }
    code_files = [
        _seal(verifier.CAMPAIGN_CONTROLLER_PATH),
        _seal(verifier.EVALUATOR_PATH),
        _seal(verifier.FOLD_VERIFIER_PATH),
        _seal(verifier.PATH_SECURITY_MODULE),
    ]
    authorization = {
        "schema_version": 1,
        "status": "authorized",
        "authorized_action": "one-time Fold 0 output recovery without scientific relaunch",
        "campaign_id": campaign_directory.name,
        "campaign_directory": str(campaign_directory.absolute()),
        "fold_id": 0,
        "run_directory": str(run.absolute()),
        "schedule": _seal(paths["schedule"]),
        "root_lock": _seal(paths["root_lock"]),
        "fold0_lock": _seal(paths["fold0_lock"]),
        "original_failure": _seal(paths["failure"]),
        "source_output": source_identity,
        "target_output": {"path": str(paths["target"].absolute()), "absent": True},
        "reviewed_code": {"commit": "f" * 40, "files": code_files},
    }
    paths["authorization"].write_text(json.dumps(authorization) + "\n", encoding="utf-8")
    resume = {
        **root_lock,
        "schema_version": 2,
        "resume_scope": "Fold 0 offline finalization then exact remaining schedule",
        "original_failure_sha256": _sha256(paths["failure"]),
        "fold0_run_directory": str(run.absolute()),
        "fold0_scientific_child_relaunched": False,
        "automatic_retry_performed": False,
        "recovery_code_commit": "f" * 40,
        "original_failure": _seal(paths["failure"]),
        "source_output": source_identity,
        "target_output": str(paths["target"].absolute()),
        "target_output_absent_before_resume": True,
    }
    paths["resume"].write_text(json.dumps(resume) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "authorized_action": "one-time atomic adoption of the existing completed child output only",
        "scientific_child_relaunched": False,
        "fold_id": 0,
        "run_directory": str(run.absolute()),
        "weight": {},
        "source": source_identity["path"],
        "target": str(paths["target"].absolute()),
        "child_pid": 1,
        "command_sha256": "a" * 64,
        "source_sha256": source_identity["sha256"],
        "source_bytes": source_identity["bytes"],
        "source_ctime_ns": source_identity["ctime_ns"],
        "source_mtime_ns": source_identity["mtime_ns"],
        "started_utc": "2026-08-12T05:00:00+00:00",
        "exit_marker_mtime_ns": 300,
        "raw_evidence_manifest": {},
        "raw_evidence_manifest_sha256": "b" * 64,
        "launch_provenance_sha256": {},
        "launch_provenance_manifest_sha256": "c" * 64,
        "external_authorization_path": str(paths["authorization"].absolute()),
        "external_authorization_sha256": _sha256(paths["authorization"]),
        "external_authorization": authorization,
    }
    paths["receipt"].write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    paths["provenance"].write_text(
        json.dumps(
            {
                "recovery_authorization_sha256": _sha256(paths["receipt"]),
                "target": str(paths["target"].absolute()),
                "sha256": source_identity["sha256"],
                "bytes": source_identity["bytes"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = {"status": "pass", "fold_count": 12, "next_fold": None}
    paths["completed"].write_text(json.dumps(completed) + "\n", encoding="utf-8")
    paths["resume_completed"].write_text(
        json.dumps(
            {
                **completed,
                "original_failure_sha256": _sha256(paths["failure"]),
                "fold0_scientific_child_relaunched": False,
                "same_campaign_directory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return state, paths


def test_campaign_recovery_chain_binds_all_incident_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    state, _ = _install_synthetic_recovery_chain(tmp_path, campaign_directory)

    verifier._verify_recovery_chain(campaign_directory, state)


@pytest.mark.parametrize("tamper", ["failure", "resume-extra", "receipt-authorization"])
def test_campaign_recovery_chain_rejects_cross_seal_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    tamper: str,
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    state, paths = _install_synthetic_recovery_chain(tmp_path, campaign_directory)
    if tamper == "failure":
        paths["failure"].write_text('{"status":"changed"}\n', encoding="utf-8")
    elif tamper == "resume-extra":
        payload = json.loads(paths["resume"].read_text(encoding="utf-8"))
        payload["unreviewed"] = True
        paths["resume"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        payload = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        payload["external_authorization"]["status"] = "changed"
        paths["receipt"].write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recovery chain"):
        verifier._verify_recovery_chain(campaign_directory, state)


def test_verifier_rejects_controller_only_summary_without_raw_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    (Path(json.loads((campaign_directory / "fold0.json").read_text())["run_directory"]) / "stdout.log").unlink()

    with pytest.raises((ValueError, FileNotFoundError)):
        verifier.verify_campaign(campaign_directory)
    assert not (campaign_directory / verifier.PUBLICATION_FILENAME).exists()


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_verifier_requires_unique_full_fold_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    case: str,
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    if case == "missing":
        (campaign_directory / "fold11.json").unlink()
    else:
        path = campaign_directory / "fold11.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fold_id"] = 10
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 0 through 11"):
        verifier.verify_campaign(campaign_directory)


def test_verifier_rejects_raw_evidence_changed_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    original = verifier.verify_fold_run

    def tamper(run: Path, weight: Path, spec: contract.WeightSpec) -> dict[str, object]:
        result = original(run, weight, spec)
        if spec.fold_id == 4:
            with (run / "stdout.log").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
        return result

    monkeypatch.setattr(verifier, "verify_fold_run", tamper)

    with pytest.raises(ValueError, match="raw evidence changed"):
        verifier.verify_campaign(campaign_directory)


def test_verifier_seals_all_raw_evidence_for_the_entire_campaign_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    original = verifier.verify_fold_run
    fold0_state = json.loads(
        (campaign_directory / "fold0.json").read_text(encoding="utf-8")
    )
    fold0_stdout = Path(fold0_state["run_directory"]) / "stdout.log"

    def tamper_earlier_fold(
        run: Path, weight: Path, spec: contract.WeightSpec
    ) -> dict[str, object]:
        result = original(run, weight, spec)
        if spec.fold_id == 11:
            with fold0_stdout.open("a", encoding="utf-8") as handle:
                handle.write("late tamper\n")
        return result

    monkeypatch.setattr(verifier, "verify_fold_run", tamper_earlier_fold)

    with pytest.raises(ValueError, match="raw evidence changed"):
        verifier.verify_campaign(campaign_directory)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_verifier_rejects_nonfinite_or_out_of_range_six_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    invalid: float,
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    state = json.loads((campaign_directory / "fold0.json").read_text(encoding="utf-8"))
    run = Path(state["run_directory"])
    metrics = {
        "test_AP": 0.1,
        "test_f1": 0.2,
        "test_iou": 0.3,
        "test_loss": invalid,
        "test_precision": 0.4,
        "test_recall": 0.5,
    }
    (run / "stdout.log").write_text(_raw_output(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match=r"six finite metrics in \[0,1\]"):
        verifier.verify_campaign(campaign_directory)


def test_verifier_rejects_generic_verifier_metric_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    original = verifier.verify_fold_run

    def disagree(run: Path, weight: Path, spec: contract.WeightSpec) -> dict[str, object]:
        result = original(run, weight, spec)
        if spec.fold_id == 6:
            result["test_metrics"] = {**result["test_metrics"], "test_AP": 0.999}
        return result

    monkeypatch.setattr(verifier, "verify_fold_run", disagree)

    with pytest.raises(ValueError, match="raw metrics disagree"):
        verifier.verify_campaign(campaign_directory)


@pytest.mark.parametrize("replace_number", [1, 2, 3, 4])
def test_publication_fault_never_commits_a_partial_generation_and_rerun_switches_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
    replace_number: int,
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    verifier.verify_campaign(campaign_directory)
    publication_path = campaign_directory / verifier.PUBLICATION_FILENAME
    previous_bytes = publication_path.read_bytes()
    previous = verifier.load_committed_publication(campaign_directory)
    real_replace = verifier.os.replace
    calls = 0

    def fail_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == replace_number:
            raise OSError(f"synthetic publication replace {replace_number}")
        real_replace(source, target)

    monkeypatch.setattr(verifier.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic publication replace"):
        verifier.verify_campaign(campaign_directory)

    assert publication_path.read_bytes() == previous_bytes
    assert verifier.load_committed_publication(campaign_directory) == previous

    monkeypatch.setattr(verifier.os, "replace", real_replace)
    result = verifier.verify_campaign(campaign_directory)
    current = verifier.load_committed_publication(campaign_directory)
    assert current["generation"] != previous["generation"]
    assert result["generation"] == current["generation"]


def test_committed_publication_rejects_generation_directory_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    publication = verifier.verify_campaign(campaign_directory)
    generation = campaign_directory / str(publication["generation"])
    alias_target = generation.with_name(generation.name + "-alias-target")
    generation.rename(alias_target)
    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(generation), str(alias_target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert created.returncode == 0, created.stdout + created.stderr
    else:
        try:
            generation.symlink_to(alias_target, target_is_directory=True)
        except OSError as error:
            if error.errno in {errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip(f"directory symlink unsupported: {error}")
            raise

    with pytest.raises(ValueError, match="generation path is aliased"):
        verifier.load_committed_publication(campaign_directory)


def test_committed_publication_rejects_hardlinked_output_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    publication = verifier.verify_campaign(campaign_directory)
    generation = campaign_directory / str(publication["generation"])
    results = generation / verifier.RESULTS_FILENAME
    summary = generation / verifier.SUMMARY_FILENAME
    summary.unlink()
    try:
        os.link(results, summary)
    except OSError as error:
        unsupported = {errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP}
        if error.errno in unsupported or getattr(error, "winerror", None) in {1, 50}:
            pytest.skip(f"hardlinks unsupported: {error}")
        raise
    shared_sha = _sha256(results)
    independent_path = generation / verifier.INDEPENDENT_FILENAME
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    independent["summary_sha256"] = shared_sha
    independent_path.write_text(
        json.dumps(independent, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    publication["outputs"][verifier.SUMMARY_FILENAME]["sha256"] = shared_sha
    publication["outputs"][verifier.INDEPENDENT_FILENAME]["sha256"] = _sha256(
        independent_path
    )
    (campaign_directory / verifier.PUBLICATION_FILENAME).write_text(
        json.dumps(publication, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publication output is aliased"):
        verifier.load_committed_publication(campaign_directory)
