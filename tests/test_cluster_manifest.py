import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cluster" / "clusterctl.py"


def load_clusterctl():
    spec = importlib.util.spec_from_file_location("clusterctl_manifest", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("clusterctl module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fixture(tmp_path: Path, years: dict[int, list[bytes]]) -> Path:
    root = tmp_path / "hdf5"
    for year, payloads in years.items():
        year_root = root / str(year)
        year_root.mkdir(parents=True)
        for index, payload in enumerate(payloads):
            (year_root / f"event-{index:03d}.hdf5").write_bytes(payload)
    return root


def output_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "manifest.csv", tmp_path / "manifest.summary.json"


def test_manifest_round_trip_records_only_direct_year_hdf5_metadata(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a", b"bb"], 2017: [b"ccc"]})
    (root / "conversion_errors.json").write_text("{}\n", encoding="utf-8")
    csv_path, summary_path = output_paths(tmp_path)

    summary = ctl.write_hdf5_manifest(root, csv_path, summary_path)

    assert summary == {
        "schema_version": 2,
        "dataset": "WSTS+ active fixed event-level HDF5",
        "file_count": 3,
        "total_bytes": 6,
        "years": {"2016": 2, "2017": 1},
    }
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "relative_path,size_bytes",
        "2016/event-000.hdf5,1",
        "2016/event-001.hdf5,2",
        "2017/event-000.hdf5,3",
    ]
    assert ctl.verify_hdf5_manifest(root, csv_path, summary_path) == summary
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("mutation", ["missing", "extra", "resize"])
def test_verify_rejects_tree_mutation(tmp_path: Path, mutation: str) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)
    ctl.write_hdf5_manifest(root, csv_path, summary_path)
    event = root / "2016" / "event-000.hdf5"

    if mutation == "missing":
        event.unlink()
    elif mutation == "extra":
        (root / "2016" / "extra.hdf5").write_bytes(b"extra")
    elif mutation == "resize":
        event.write_bytes(b"aa")
    with pytest.raises(ValueError):
        ctl.verify_hdf5_manifest(root, csv_path, summary_path)


def test_verify_intentionally_ignores_same_size_content_changes(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)
    ctl.write_hdf5_manifest(root, csv_path, summary_path)

    (root / "2016" / "event-000.hdf5").write_bytes(b"b")

    assert ctl.verify_hdf5_manifest(root, csv_path, summary_path)["file_count"] == 1


@pytest.mark.parametrize("unsafe", ["nested", "non-year", "wrong-extension"])
def test_builder_rejects_noncanonical_tree_layout(tmp_path: Path, unsafe: str) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    if unsafe == "nested":
        nested = root / "2016" / "nested"
        nested.mkdir()
        (nested / "event.hdf5").write_bytes(b"nested")
    elif unsafe == "non-year":
        (root / "latest").mkdir()
    else:
        (root / "2016" / "notes.txt").write_text("notes", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        ctl.build_hdf5_manifest(root)


def test_builder_rejects_symlinked_event_when_supported(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    source = tmp_path / "outside.hdf5"
    source.write_bytes(b"outside")
    link = root / "2016" / "link.hdf5"
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlink capability unavailable: {error}")

    with pytest.raises(ValueError, match="link"):
        ctl.build_hdf5_manifest(root)


@pytest.mark.parametrize("tamper", ["duplicate", "unsafe-path", "invalid-size"])
def test_verifier_rejects_malformed_manifest_rows(tmp_path: Path, tamper: str) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)
    ctl.write_hdf5_manifest(root, csv_path, summary_path)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    if tamper == "duplicate":
        rows.append(dict(rows[0]))
    elif tamper == "unsafe-path":
        rows[0]["relative_path"] = "../outside.hdf5"
    else:
        rows[0]["size_bytes"] = "01"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("relative_path", "size_bytes")
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError):
        ctl.verify_hdf5_manifest(root, csv_path, summary_path)


def test_verifier_rejects_wrong_summary_metadata(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)
    ctl.write_hdf5_manifest(root, csv_path, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["file_count"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary"):
        ctl.verify_hdf5_manifest(root, csv_path, summary_path)


def test_manifest_cli_create_and_verify_fixture(tmp_path: Path) -> None:
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)

    created = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "manifest",
            "create",
            str(root),
            str(csv_path),
            str(summary_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "manifest",
            "verify",
            str(root),
            str(csv_path),
            str(summary_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    assert verified.returncode == 0, verified.stderr


def test_production_contract_rejects_fixture_summary(tmp_path: Path) -> None:
    ctl = load_clusterctl()
    root = make_fixture(tmp_path, {2016: [b"a"]})
    csv_path, summary_path = output_paths(tmp_path)

    with pytest.raises(ValueError, match="production dataset contract"):
        ctl.write_hdf5_manifest(
            root,
            csv_path,
            summary_path,
            require_production_contract=True,
        )
    assert not csv_path.exists()
    assert not summary_path.exists()


def test_tracked_manifest_outputs_are_not_ignored() -> None:
    paths = [
        "manifests/data/wstsplus-hdf5.csv",
        "manifests/data/wstsplus-hdf5.summary.json",
    ]

    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, path


def test_heavy_cluster_roots_are_ignored_without_hiding_weight_manifest() -> None:
    for path in ("weights/example.pth", "scratch/tmp.bin"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, path
    manifest = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            "manifests/weights/README.md",
        ],
        cwd=ROOT,
        check=False,
    )
    assert manifest.returncode == 1
