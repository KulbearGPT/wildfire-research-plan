import json

import pytest

from reproductions.cross_history.compose_table1 import compose_table, render_markdown


def _summary(path, *, method, seed, year, offset):
    results = {
        "M00": {"avg_precision": 0.50 + offset},
        "M01": {"avg_precision": 0.20 + offset},
        "M06": {"avg_precision": 0.30 + offset},
        "M07": {"avg_precision": 0.10 + offset},
    }
    path.write_text(json.dumps({
        "architecture": "swin_unet", "history": 1, "method": method,
        "seed": seed, "year": year, "results": results,
    }))
    return path


def test_table_uses_matched_fixed_test_years_and_excludes_selection_year(tmp_path):
    paths = []
    for seed in (0, 1, 2):
        for year in (2022, 2023):
            paths.append(_summary(tmp_path / f"c-{seed}-{year}.json",
                                  method="control", seed=seed, year=year,
                                  offset=seed * 0.01))
            paths.append(_summary(tmp_path / f"x-{seed}-{year}.json",
                                  method="cosine_erm", seed=seed, year=year,
                                  offset=0.02 + seed * 0.01))
    paths.append(_summary(tmp_path / "selection.json", method="cosine_erm",
                          seed=0, year=2021, offset=0.40))

    table = compose_table(paths)

    control, cosine = table["rows"]
    assert (control["architecture"], control["history"], control["method"]) == (
        "swin_unet", 1, "control")
    assert control["M00"] == pytest.approx(0.51)
    assert control["primary"] == pytest.approx(0.21)
    assert cosine["primary"] == pytest.approx(0.23)
    assert cosine["delta"] == pytest.approx(0.02)
    assert cosine["seed_std"] == pytest.approx(0.00816496580927726)
    assert table["years"] == [2022, 2023]
    markdown = render_markdown(table)
    assert "| swin_unet | 1 | cosine_erm |" in markdown
    assert "0.230000 ± 0.008165" in markdown
    assert "+0.020000" in markdown


def test_table_rejects_incomplete_or_unmatched_test_cells(tmp_path):
    paths = []
    for seed in (0, 1, 2):
        for method in ("control", "cosine_erm"):
            paths.append(_summary(tmp_path / f"{method}-{seed}.json",
                                  method=method, seed=seed, year=2022,
                                  offset=0.0 if method == "control" else 0.02))

    with pytest.raises(ValueError, match="2022 and 2023"):
        compose_table(paths)
