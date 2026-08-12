"""Frozen metadata for the official twelve Res18-U-Net T=1 weights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
REPO_ID = "saadlahrichi/WSTSPlus"
REVISION = "acf70a37394849f4ec8d108a51d6f4325a554d0a"
WEIGHT_PREFIX = "trained_model_weights/Res18Unet_T1/All/"
RELEASED_WEIGHT_FILENAMES = (
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
)
OFFICIAL_FOLDS = (
    (2018, 2019, 2020, 2021),
    (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021),
    (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020),
    (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021),
    (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020),
    (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019),
    (2020, 2021, 2019, 2018),
)
_FILENAME_PATTERN = re.compile(r"fold(\d+)_testAP(0\.\d+)\.pth")


@dataclass(frozen=True, slots=True)
class WeightSpec:
    fold_id: int
    filename: str
    hub_path: str
    size: int
    sha256: str
    filename_ap: float
    train_years: tuple[int, int]
    validation_year: int
    test_year: int


def _filename_metadata(filename: str) -> tuple[int, float]:
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError("released weight filenames differ from the frozen contract")
    fold_id = int(match.group(1))
    if not 0 <= fold_id < len(OFFICIAL_FOLDS):
        raise ValueError("released weight fold is outside 0 through 11")
    if filename != RELEASED_WEIGHT_FILENAMES[fold_id]:
        raise ValueError("released weight filenames differ from the frozen contract")
    return fold_id, float(match.group(2))


def _weight_spec(path: str, size: object, sha256: object) -> WeightSpec:
    if not path.startswith(WEIGHT_PREFIX):
        raise ValueError("released weight path is outside the frozen prefix")
    filename = path.removeprefix(WEIGHT_PREFIX)
    fold_id, filename_ap = _filename_metadata(filename)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("released weight size must be positive")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("released weight SHA-256 must be 64 lowercase hexadecimal characters")
    train_a, train_b, validation_year, test_year = OFFICIAL_FOLDS[fold_id]
    return WeightSpec(
        fold_id=fold_id,
        filename=filename,
        hub_path=path,
        size=size,
        sha256=sha256,
        filename_ap=filename_ap,
        train_years=(train_a, train_b),
        validation_year=validation_year,
        test_year=test_year,
    )


def parse_pinned_tree(items: Sequence[Mapping[str, object]]) -> tuple[WeightSpec, ...]:
    """Parse the exact All/T=1 entries returned by the pinned Hub tree."""
    weights = [
        item
        for item in items
        if item.get("type") == "file"
        and isinstance(item.get("path"), str)
        and str(item["path"]).startswith(WEIGHT_PREFIX)
    ]
    if len(weights) != len(OFFICIAL_FOLDS):
        raise ValueError("pinned tree must contain exactly twelve released weights")
    specs: list[WeightSpec] = []
    for item in weights:
        lfs = item.get("lfs")
        sha256 = lfs.get("oid") if isinstance(lfs, Mapping) else None
        specs.append(_weight_spec(str(item["path"]), item.get("size"), sha256))
    ordered = tuple(sorted(specs, key=lambda spec: spec.fold_id))
    if len({spec.fold_id for spec in ordered}) != len(OFFICIAL_FOLDS):
        raise ValueError("released weight filenames do not contain unique folds")
    if tuple(spec.filename for spec in ordered) != RELEASED_WEIGHT_FILENAMES:
        raise ValueError("released weight filenames differ from the frozen contract")
    return ordered


def spec_for_fold(specs: Sequence[WeightSpec], fold_id: int) -> WeightSpec:
    selected = [spec for spec in specs if spec.fold_id == fold_id]
    if len(selected) != 1:
        raise ValueError(f"fold {fold_id} weight is missing or ambiguous")
    return selected[0]


def _manifest_payload(specs: Sequence[WeightSpec]) -> dict[str, object]:
    ordered = tuple(sorted(specs, key=lambda spec: spec.fold_id))
    if len(ordered) != len(OFFICIAL_FOLDS) or tuple(
        spec.filename for spec in ordered
    ) != RELEASED_WEIGHT_FILENAMES:
        raise ValueError("manifest must contain exactly the frozen twelve weights")
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "weight_prefix": WEIGHT_PREFIX,
        "weights": [asdict(spec) for spec in ordered],
    }


def write_manifest_atomic(path: Path, specs: Sequence[WeightSpec]) -> None:
    """Atomically publish the schema-versioned pinned weight manifest."""
    payload = _manifest_payload(specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pinned_manifest(path: Path) -> tuple[WeightSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("pinned manifest must be a JSON object")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "weight_prefix": WEIGHT_PREFIX,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("pinned manifest header differs from the frozen contract")
    rows = payload.get("weights")
    if not isinstance(rows, list):
        raise ValueError("pinned manifest weights must be a list")
    tree: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pinned manifest weight row must be an object")
        tree.append(
            {
                "type": "file",
                "path": row.get("hub_path"),
                "size": row.get("size"),
                "lfs": {"oid": row.get("sha256")},
            }
        )
    specs = parse_pinned_tree(tree)
    expected_rows = json.loads(json.dumps([asdict(spec) for spec in specs]))
    if rows != expected_rows:
        raise ValueError("pinned manifest metadata differs from the frozen contract")
    return specs


def validate_local_weight(path: Path, spec: WeightSpec) -> None:
    if not path.is_file() or path.stat().st_size != spec.size:
        raise ValueError("released weight download size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ValueError("released weight download SHA-256 mismatch")
