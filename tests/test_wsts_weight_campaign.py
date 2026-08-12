import sys
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import released_weight_contract as contract  # noqa: E402
import run_weight_campaign as campaign  # noqa: E402


@pytest.fixture(scope="module")
def specs() -> tuple[contract.WeightSpec, ...]:
    return contract.load_pinned_manifest(
        REPOSITORY_ROOT
        / "reproductions"
        / "wsts_res18_unet_t1"
        / "official_weights_manifest.json"
    )


def test_schedule_is_exact_and_adopts_only_fold2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    adopted = (tmp_path / "fold2-weight-existing").resolve()
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", adopted)

    schedule = campaign.build_campaign_schedule(specs, {2: adopted})

    assert tuple(item.spec.fold_id for item in schedule) == tuple(range(12))
    assert tuple(item.mode for item in schedule) == (
        "launch", "launch", "adopt", "launch", "launch", "launch",
        "launch", "launch", "launch", "launch", "launch", "launch",
    )
    assert schedule[2].run_directory == adopted
    assert all(item.run_directory is None for item in schedule if item.spec.fold_id != 2)


@pytest.mark.parametrize("adopted", [{}, {1: Path("wrong")}, {2: Path("a"), 3: Path("b")}])
def test_schedule_rejects_any_adoption_other_than_exact_fold2(
    specs: tuple[contract.WeightSpec, ...], adopted: dict[int, Path]
) -> None:
    with pytest.raises(ValueError, match="exactly Fold 2"):
        campaign.build_campaign_schedule(specs, adopted)


def test_qualify_fold2_requires_generic_pass_before_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 2)
    run = (tmp_path / "fold2-weight-sealed").resolve()
    run.mkdir()
    independent = run / "independent-verification.json"
    independent.write_text('{"status":"pass"}\n', encoding="utf-8")
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", run)
    monkeypatch.setattr(campaign, "weight_cache_path", lambda _spec: tmp_path / spec.filename)
    monkeypatch.setattr(
        campaign,
        "verify_fold_run",
        lambda *_args: {"status": "fail", "fold_id": 2},
    )

    with pytest.raises(ValueError, match="generic independent verifier did not pass"):
        campaign.qualify_existing_fold2(run, spec)


def _configure_fake_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verifier_failure: int | None = None,
    child_failure: int | None = None,
) -> tuple[Path, list[int], list[int]]:
    root = tmp_path / "campaigns"
    campaign_directory = root / "official-weight-12fold-test"
    fold_runs = tmp_path / "fold-runs"
    adopted = fold_runs / "fold2-weight-adopted"
    adopted.mkdir(parents=True)
    (adopted / "independent-verification.json").write_text(
        '{"status":"pass","fold_id":2}\n', encoding="utf-8"
    )
    scientific_folds: list[int] = []
    qualified_folds: list[int] = []

    monkeypatch.setattr(campaign, "CAMPAIGN_ARTIFACTS_ROOT", root)
    monkeypatch.setattr(campaign, "ADOPTED_FOLD2_RUN", adopted.resolve())
    monkeypatch.setattr(campaign, "prepare_weight", lambda _spec: {"status": "pass"})
    def qualify(run: Path, spec: contract.WeightSpec) -> dict[str, object]:
        qualified_folds.append(spec.fold_id)
        return {
            "status": "pass",
            "fold_id": spec.fold_id,
            "run_directory": str(run.resolve()),
            "independent_verification_sha256": campaign.sha256_file(
                run / "independent-verification.json"
            ),
        }

    monkeypatch.setattr(campaign, "qualify_existing_fold2", qualify)

    def launch(spec: contract.WeightSpec) -> tuple[Path, dict[str, object]]:
        scientific_folds.append(spec.fold_id)
        if spec.fold_id == child_failure:
            raise RuntimeError("synthetic child failure")
        run = fold_runs / f"fold{spec.fold_id}-weight-launched"
        run.mkdir(parents=True)
        return run, {"status": "pass"}

    def verify(run: Path, spec: contract.WeightSpec) -> dict[str, object]:
        if spec.fold_id == verifier_failure:
            return {"status": "fail", "fold_id": spec.fold_id}
        path = run / "independent-verification.json"
        path.write_text(
            json.dumps({"status": "pass", "fold_id": spec.fold_id}) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "pass",
            "fold_id": spec.fold_id,
            "run_directory": str(run.resolve()),
            "independent_verification_sha256": campaign.sha256_file(path),
        }

    monkeypatch.setattr(campaign, "launch_one_fold_once", launch)
    monkeypatch.setattr(campaign, "verify_one_fold", verify)
    return campaign_directory, scientific_folds, qualified_folds


def test_campaign_launches_eleven_folds_sequentially_and_records_no_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, qualified_folds = _configure_fake_campaign(
        tmp_path, monkeypatch
    )

    result = campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert qualified_folds == [2]
    assert result == {"status": "pass", "fold_count": 12, "next_fold": None}
    schedule = json.loads((campaign_directory / "schedule.json").read_text(encoding="utf-8"))
    assert [row["fold_id"] for row in schedule["folds"]] == list(range(12))
    assert schedule["folds"][2]["mode"] == "adopt"
    for fold_id in range(12):
        state = json.loads(
            (campaign_directory / f"fold{fold_id}.json").read_text(encoding="utf-8")
        )
        assert set(state) == {
            "fold_id", "independent_verification_sha256", "mode", "run_directory", "status"
        }


def test_campaign_stops_before_next_fold_when_verifier_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, verifier_failure=4
    )

    with pytest.raises(ValueError, match="fold 4 independent verification failed"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4]
    assert not (campaign_directory / "fold5.json").exists()
    failure = json.loads((campaign_directory / "failure.json").read_text(encoding="utf-8"))
    assert failure["fold_id"] == 4
    assert failure["automatic_retry_performed"] is False


def test_campaign_stops_before_verifier_and_next_fold_when_child_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _ = _configure_fake_campaign(
        tmp_path, monkeypatch, child_failure=3
    )

    with pytest.raises(RuntimeError, match="synthetic child failure"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3]
    assert not (campaign_directory / "fold3.json").exists()
    assert not (campaign_directory / "fold4.lock.json").exists()


@pytest.mark.parametrize("lock_kind", ["global", "campaign", "fold"])
def test_existing_immutable_lock_refuses_campaign_before_any_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_kind: str
) -> None:
    campaign_directory, scientific_folds, _ = _configure_fake_campaign(
        tmp_path, monkeypatch
    )
    campaign_directory.mkdir(parents=True)
    if lock_kind == "global":
        lock = campaign.CAMPAIGN_ARTIFACTS_ROOT / campaign.GLOBAL_LOCK_NAME
    elif lock_kind == "campaign":
        lock = campaign_directory / "campaign.lock.json"
    else:
        lock = campaign_directory / "fold0.lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="lock already exists"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == []
    assert lock.read_text(encoding="utf-8") == "sentinel\n"


def test_second_campaign_call_cannot_duplicate_a_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_directory, scientific_folds, _ = _configure_fake_campaign(
        tmp_path, monkeypatch
    )
    campaign.run_campaign(campaign_directory)

    with pytest.raises(FileExistsError, match="lock already exists"):
        campaign.run_campaign(campaign_directory)

    assert scientific_folds == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_finalize_existing_fold_never_launches_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[contract.WeightSpec, ...],
) -> None:
    spec = contract.spec_for_fold(specs, 5)
    run = tmp_path / "fold5-weight-existing"
    run.mkdir()
    monkeypatch.setattr(
        campaign,
        "launch_one_fold_once",
        lambda _spec: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    monkeypatch.setattr(
        campaign,
        "finalize_existing_weight",
        lambda selected, selected_spec: {"status": "pass", "fold_id": selected_spec.fold_id},
    )
    monkeypatch.setattr(
        campaign,
        "verify_one_fold",
        lambda selected, selected_spec: {
            "status": "pass", "fold_id": selected_spec.fold_id, "run_directory": str(selected)
        },
    )

    result = campaign.finalize_existing_fold(run, spec)

    assert result["status"] == "pass"
    assert result["fold_id"] == 5
