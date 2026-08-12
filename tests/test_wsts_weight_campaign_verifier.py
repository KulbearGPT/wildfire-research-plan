import sys
import csv
import hashlib
import json
import statistics
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
    summary = json.loads(
        (campaign_directory / verifier.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
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
    with (campaign_directory / verifier.RESULTS_FILENAME).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["fold_id"]) for row in rows] == list(range(12))
    assert [int(row["test_year"]) for row in rows] == [spec.test_year for spec in specs]
    assert float(rows[0]["test_AP"]) == values[0]
    assert float(rows[0]["test_AP"]) != 0.999
    assert float(rows[2]["test_AP_minus_filename_ap"]) == values[2] - specs[2].filename_ap
    assert not list(campaign_directory.glob("*.tmp"))


def test_verifier_rejects_controller_only_summary_without_raw_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    campaign_directory, _ = _build_synthetic_campaign(tmp_path, monkeypatch, specs)
    (Path(json.loads((campaign_directory / "fold0.json").read_text())["run_directory"]) / "stdout.log").unlink()

    with pytest.raises((ValueError, FileNotFoundError)):
        verifier.verify_campaign(campaign_directory)
    assert not (campaign_directory / verifier.RESULTS_FILENAME).exists()


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
