from pathlib import Path

from wildfire_phase0.contract import ContractDecision
from wildfire_phase0.report import render_phase0_report
from wildfire_phase0.schema import EventInventory
from wildfire_phase0.splits import build_forward_split


def test_report_uses_pixel_weighted_dataset_nan_fraction_and_event_maximum() -> None:
    inventory = [
        EventInventory(
            2021,
            "small_missing",
            Path("2021/small_missing.hdf5"),
            1,
            23,
            1,
            1,
            ("2021-08-01",),
            1.0,
            1,
            1,
            1,
            9,
            9,
        ),
        EventInventory(
            2021,
            "large_complete",
            Path("2021/large_complete.hdf5"),
            1,
            23,
            3,
            3,
            ("2021-08-01",),
            0.0,
            2,
            0,
            3,
            6,
            9,
        ),
    ]
    decision = ContractDecision("continue_controlled", (), (), ())

    report = render_phase0_report(
        inventory,
        build_forward_split(inventory),
        decision,
        None,
        (),
    )

    assert "Pixel-weighted dataset NaN fraction: 0.1" in report
    assert "Maximum event NaN fraction: 1" in report
    assert "Sum of event NaN fractions" not in report
    assert (
        "| year | events | target days | zero-target days | positive target pixels |"
        in report
    )
    assert "| 2021 | 2 | 3 | 1 | 4 |" in report
    assert (
        "| split | events | target days | zero-target days | positive target pixels |"
        in report
    )
    assert "| validation | 2 | 3 | 1 | 4 |" in report
