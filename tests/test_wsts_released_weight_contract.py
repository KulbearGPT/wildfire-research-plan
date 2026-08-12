import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproductions" / "wsts_res18_unet_t1" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import released_weight_contract as contract  # noqa: E402


FILENAMES = (
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


def _tree_items() -> list[dict[str, object]]:
    return [
        {
            "type": "file",
            "path": contract.WEIGHT_PREFIX + filename,
            "size": fold_id + 1,
            "lfs": {"oid": f"{fold_id:x}" * 64},
        }
        for fold_id, filename in enumerate(FILENAMES)
    ]


def test_contract_freezes_official_twelve_folds() -> None:
    assert contract.OFFICIAL_FOLDS == (
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
    assert contract.REVISION == "acf70a37394849f4ec8d108a51d6f4325a554d0a"
    assert contract.REPO_ID == "saadlahrichi/WSTSPlus"
    assert contract.WEIGHT_PREFIX == "trained_model_weights/Res18Unet_T1/All/"


def test_pinned_tree_maps_each_filename_to_its_fold_metadata() -> None:
    specs = contract.parse_pinned_tree(list(reversed(_tree_items())))

    assert tuple(spec.fold_id for spec in specs) == tuple(range(12))
    assert tuple(spec.filename for spec in specs) == FILENAMES
    assert specs[2].filename_ap == 0.571
    assert specs[2].train_years == (2018, 2020)
    assert specs[2].validation_year == 2019
    assert specs[2].test_year == 2021


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda items: items.pop(), "exactly twelve"),
        (lambda items: items.append(dict(items[0])), "exactly twelve"),
        (lambda items: items.__setitem__(0, dict(items[0], path=contract.WEIGHT_PREFIX + "renamed.pth")), "filenames"),
        (lambda items: items.__setitem__(0, dict(items[0], size=0)), "positive"),
        (lambda items: items.__setitem__(0, dict(items[0], lfs={"oid": "A" * 64})), "SHA-256"),
    ],
)
def test_pinned_tree_fails_closed_on_invalid_entries(mutate, message: str) -> None:
    items = _tree_items()
    mutate(items)

    with pytest.raises(ValueError, match=message):
        contract.parse_pinned_tree(items)


def test_manifest_round_trip_is_schema_versioned_and_fold_sorted(tmp_path: Path) -> None:
    manifest = tmp_path / "official_weights_manifest.json"
    contract.write_manifest_atomic(manifest, tuple(reversed(contract.parse_pinned_tree(_tree_items()))))

    loaded = contract.load_pinned_manifest(manifest)

    assert tuple(spec.fold_id for spec in loaded) == tuple(range(12))
    assert loaded == contract.parse_pinned_tree(_tree_items())


def test_local_weight_rejects_wrong_size_and_hash(tmp_path: Path) -> None:
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"official")
    digest = hashlib.sha256(b"official").hexdigest()
    spec = contract.WeightSpec(0, FILENAMES[0], contract.WEIGHT_PREFIX + FILENAMES[0], 8, digest, 0.528, (2018, 2019), 2020, 2021)

    contract.validate_local_weight(weight, spec)
    with pytest.raises(ValueError, match="size"):
        contract.validate_local_weight(weight, replace(spec, size=9))
    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate_local_weight(weight, replace(spec, sha256="0" * 64))
