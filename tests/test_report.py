from pathlib import Path

import pandas as pd

from wildfire_phase0.contract import ContractDecision
from wildfire_phase0.report import render_phase0_report
from wildfire_phase0.schema import EventInventory


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
        ),
    ]
    decision = ContractDecision("continue_controlled", (), (), ())

    report = render_phase0_report(
        inventory,
        pd.DataFrame(columns=["split"]),
        decision,
        None,
        (),
    )

    assert "Pixel-weighted dataset NaN fraction: 0.1" in report
    assert "Maximum event NaN fraction: 1" in report
    assert "Sum of event NaN fractions" not in report
