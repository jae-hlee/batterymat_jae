import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from batterymat.screening_cathode.screen_cathode import (
    extract_jid,
    load_and_merge,
    filter_cathode_candidates,
    rank_candidates,
    run_screening,
)


def test_extract_jid_standard():
    assert extract_jid("Li_JVASP-47127_Li3SiNi3O8.json") == "JVASP-47127"


def test_extract_jid_short_name():
    assert extract_jid("Li_JVASP-1_LiO.json") == "JVASP-1"


def test_extract_jid_malformed():
    with pytest.raises(ValueError):
        extract_jid("malformed")


def make_li_min_df(overrides=None):
    base = {
        "name": ["Li_JVASP-100_LiMnO2.json", "Li_JVASP-200_LiCoO2.json"],
        "avg_voltage": [3.5, 4.0],
        "max_voltage": [4.0, 4.5],
        "max_grav_cap": [120.0, 150.0],
        "max_vol_cap": [400.0, 500.0],
        "voltage_profile": ["3.5;4.0", "4.0;4.5"],
        "grav_capacity_profile": ["120.0", "150.0"],
        "vol_capacity_profile": ["400.0", "500.0"],
    }
    if overrides:
        base.update(overrides)
    return pd.DataFrame(base)


def make_dft3d_df():
    return pd.DataFrame({
        "jid": ["JVASP-100", "JVASP-200"],
        "ehull": [0.02, 0.04],
        "optb88vdw_bandgap": [1.5, 0.0],
        "formula": ["LiMnO2", "LiCoO2"],
        "atoms": [{}, {}],
    })


def test_load_and_merge_adds_ehull():
    li_df = make_li_min_df()
    dft_df = make_dft3d_df()
    result = load_and_merge(li_df, dft_df)
    assert "ehull" in result.columns
    assert "jid" in result.columns
    assert len(result) == 2
    row_100 = result[result["jid"] == "JVASP-100"]
    assert len(row_100) == 1
    assert row_100.iloc[0]["ehull"] == 0.02


def test_load_and_merge_drops_missing_jids(capsys):
    li_df = make_li_min_df(overrides={
        "name": ["Li_JVASP-100_LiMnO2.json", "Li_JVASP-999_Unknown.json"],
        "avg_voltage": [3.5, 3.8],
        "max_voltage": [4.0, 4.2],
        "max_grav_cap": [120.0, 110.0],
        "max_vol_cap": [400.0, 380.0],
        "voltage_profile": ["3.5;4.0", "3.8;4.2"],
        "grav_capacity_profile": ["120.0", "110.0"],
        "vol_capacity_profile": ["400.0", "380.0"],
    })
    dft_df = make_dft3d_df()  # only has JVASP-100 and JVASP-200
    result = load_and_merge(li_df, dft_df)
    captured = capsys.readouterr()
    assert len(result) == 1
    assert "JVASP-999" in captured.out


def test_load_and_merge_raises_on_empty_dft3d():
    li_df = make_li_min_df()
    with pytest.raises(RuntimeError, match="JARVIS-DFT"):
        load_and_merge(li_df, None)


def make_merged_df(rows):
    """Helper: build a merged DataFrame directly (bypassing load_and_merge)."""
    return pd.DataFrame(rows)


def test_filter_removes_low_avg_voltage():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 2.9, "max_voltage": 3.5, "max_grav_cap": 120.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_filter_removes_high_avg_voltage():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 4.6, "max_voltage": 5.0, "max_grav_cap": 120.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_filter_removes_low_capacity():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 19.9, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 20.1, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_filter_removes_high_ehull():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.06},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.04},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_filter_removes_high_max_voltage():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 5.6, "max_grav_cap": 120.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_filter_boundary_avg_voltage_inclusive():
    """avg_voltage == 3.0 and == 4.5 should both pass (inclusive boundaries)."""
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.0, "max_voltage": 4.0, "max_grav_cap": 120.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 4.5, "max_voltage": 5.0, "max_grav_cap": 120.0, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert set(result["jid"]) == {"A", "B"}


def test_filter_boundary_max_grav_cap_strict():
    """max_grav_cap == 20.0 should fail (strict >); 20.1 should pass."""
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 20.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 20.1, "ehull": 0.02},
    ])
    result = filter_cathode_candidates(df)
    assert list(result["jid"]) == ["B"]


def test_rank_score_formula():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.0, "max_grav_cap": 100.0, "ehull": 0.05},
        {"jid": "B", "avg_voltage": 4.5, "max_grav_cap": 200.0, "ehull": 0.00},
    ])
    result = rank_candidates(df)
    # B should score higher: max voltage, max cap, min ehull
    assert result.iloc[0]["jid"] == "B"
    assert result.iloc[1]["jid"] == "A"


def test_rank_constant_column_returns_half():
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_grav_cap": 120.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_grav_cap": 150.0, "ehull": 0.02},
    ])
    # avg_voltage and ehull are constant — their norm should be 0.5
    result = rank_candidates(df)
    # Score = (1/3)*0.5 + (1/3)*norm(cap) - (1/3)*0.5
    # For B: cap_norm = 1.0 → score = 0.5/3 + 1/3 - 0.5/3 = 1/3
    # For A: cap_norm = 0.0 → score = 0.5/3 + 0   - 0.5/3 = 0
    assert result.iloc[0]["jid"] == "B"


def test_rank_fewer_than_two_candidates_returns_unranked(capsys):
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_grav_cap": 120.0, "ehull": 0.02},
    ])
    result = rank_candidates(df)
    captured = capsys.readouterr()
    assert len(result) == 1
    assert "WARNING" in captured.out
    assert "score" not in result.columns


import os
import tempfile


def test_run_screening_writes_csv(tmp_path):
    li_min_df = make_li_min_df(overrides={
        "avg_voltage": [3.5, 2.0],       # second candidate below filter
        "max_voltage": [4.0, 3.0],
        "max_grav_cap": [120.0, 80.0],
    })
    dft_df = make_dft3d_df()
    output_path = tmp_path / "candidates.csv"
    result = run_screening(
        li_min_df=li_min_df,
        dft3d_df=dft_df,
        output_path=str(output_path),
    )
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == 1
    assert saved.iloc[0]["jid"] == "JVASP-100"
    assert "score" in saved.columns


def test_run_screening_raises_on_fetch_failure():
    """run_screening raises RuntimeError when dft3d_df=None and fetch fails."""
    li_min_df = make_li_min_df()
    with patch("jarvis.db.figshare.data", side_effect=Exception("network error")):
        with pytest.raises(RuntimeError, match="Failed to fetch JARVIS-DFT"):
            run_screening(li_min_df=li_min_df, dft3d_df=None)


def test_run_screening_output_includes_bandgap(tmp_path):
    """optb88vdw_bandgap must appear in the saved CSV (spec Section 3 Output)."""
    li_min_df = make_li_min_df()
    dft_df = make_dft3d_df()
    output_path = tmp_path / "out.csv"
    run_screening(li_min_df=li_min_df, dft3d_df=dft_df, output_path=str(output_path))
    saved = pd.read_csv(output_path)
    assert "optb88vdw_bandgap" in saved.columns
