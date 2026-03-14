# Cathode Screening + DFT Validation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Li-ion cathode screening module that filters and ranks ML-screened candidates, then generates complete VASP input directories for DFT validation.

**Architecture:** Two independent stages — Stage 1 (`screen_cathode.py`) does pure pandas data processing on existing `Li_min.csv` + JARVIS-DFT metadata; Stage 2 (`dft_prep.py`) takes user-selected JIDs, generates the lowest-energy delithiated structure via ALIGNN-ranked Wyckoff vacancies, and writes 4 VASP input directories per candidate. A Jupyter notebook ties both stages together for interactive use.

**Tech Stack:** Python 3.8+, pandas, numpy, jarvis-tools (`Atoms`, `Vacancy`, `Spacegroup3D`, JARVIS-DFT figshare), alignn (`AlignnAtomwiseCalculator`, `wt01_path`), matplotlib, nbformat (notebook creation)

**Spec:** `docs/superpowers/specs/2026-03-11-cathode-screening-dft-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `batterymat/screening_cathode/__init__.py` | Create | Module init (empty) |
| `batterymat/screening_cathode/screen_cathode.py` | Create | Load/merge/filter/rank/export candidates |
| `batterymat/screening_cathode/dft_prep.py` | Create | Vacancy selection, VASP input gen, voltage helper |
| `batterymat/screening_cathode/cathode_filtering.ipynb` | Create | Interactive notebook (explore + DFT export) |
| `tests/__init__.py` | Create | Test package init |
| `tests/test_screen_cathode.py` | Create | Unit tests for screening logic |
| `tests/test_dft_prep.py` | Create | Unit tests for DFT prep logic |

---

## Chunk 1: screen_cathode.py

### Task 1: Module scaffold

**Files:**
- Create: `batterymat/screening_cathode/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the module directory and init files**

```bash
mkdir -p batterymat/screening_cathode
touch batterymat/screening_cathode/__init__.py
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Verify import works**

```bash
python -c "import batterymat.screening_cathode; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add batterymat/screening_cathode/__init__.py tests/__init__.py
git commit -m "feat: scaffold screening_cathode module and tests directory"
```

---

### Task 2: JID extraction and dataset merge

**Files:**
- Create: `batterymat/screening_cathode/screen_cathode.py`
- Create: `tests/test_screen_cathode.py`

**Context:** `Li_min.csv` has a `name` column like `Li_JVASP-47127_Li3SiNi3O8.json`. The JID is the second `_`-delimited token. The file has no `jid`, `ehull`, or `optb88vdw_bandgap` columns — those come from JARVIS-DFT via an inner join on `jid`.

- [ ] **Step 1: Write failing tests for JID extraction and merge**

Create `tests/test_screen_cathode.py`:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from batterymat.screening_cathode.screen_cathode import (
    extract_jid,
    load_and_merge,
)


def test_extract_jid_standard():
    assert extract_jid("Li_JVASP-47127_Li3SiNi3O8.json") == "JVASP-47127"


def test_extract_jid_short_name():
    assert extract_jid("Li_JVASP-1_LiO.json") == "JVASP-1"


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_screen_cathode.py::test_extract_jid_standard tests/test_screen_cathode.py::test_load_and_merge_adds_ehull -v
```
Expected: `ImportError` or `ModuleNotFoundError` (screen_cathode.py doesn't exist yet)

- [ ] **Step 3: Implement `extract_jid` and `load_and_merge` in screen_cathode.py**

Create `batterymat/screening_cathode/screen_cathode.py`:

```python
"""Cathode candidate screening from Li_min.csv + JARVIS-DFT metadata."""
import sys
import warnings
from typing import Optional
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Filtering thresholds (spec Section 3)
# ---------------------------------------------------------------------------
FILTERS = {
    "avg_voltage_min": 3.0,
    "avg_voltage_max": 4.5,
    "max_grav_cap_min": 100.0,
    "ehull_max": 0.05,
    "max_voltage_max": 5.5,
}


def extract_jid(name: str) -> str:
    """Extract JARVIS JID from a Li_min.csv name field.

    Example: 'Li_JVASP-47127_Li3SiNi3O8.json' -> 'JVASP-47127'
    """
    return name.split("_")[1]


def load_and_merge(
    li_min_df: pd.DataFrame,
    dft3d_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge Li_min.csv DataFrame with JARVIS-DFT metadata on JID.

    Args:
        li_min_df: DataFrame loaded from Li_min.csv.
        dft3d_df: DataFrame from jarvis.db.figshare.data('dft_3d').
                  Pass None to trigger a RuntimeError (offline guard).

    Returns:
        Merged DataFrame with ehull, optb88vdw_bandgap, formula, atoms added.
    """
    if dft3d_df is None:
        raise RuntimeError(
            "JARVIS-DFT dataset is None. Fetch it with "
            "jarvis.db.figshare.data('dft_3d') or provide a cached copy."
        )

    li_min_df = li_min_df.copy()
    li_min_df["jid"] = li_min_df["name"].apply(extract_jid)

    merged = pd.merge(li_min_df, dft3d_df[["jid", "ehull", "optb88vdw_bandgap", "formula", "atoms"]], on="jid", how="inner")

    missing_jids = set(li_min_df["jid"]) - set(merged["jid"])
    if missing_jids:
        print(
            f"WARNING: {len(missing_jids)} JIDs from Li_min.csv not found in "
            f"JARVIS-DFT dataset and were dropped: {sorted(missing_jids)}",
            file=sys.stdout,
        )

    return merged.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_screen_cathode.py::test_extract_jid_standard tests/test_screen_cathode.py::test_extract_jid_short_name tests/test_screen_cathode.py::test_load_and_merge_adds_ehull tests/test_screen_cathode.py::test_load_and_merge_drops_missing_jids tests/test_screen_cathode.py::test_load_and_merge_raises_on_empty_dft3d -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/screen_cathode.py tests/test_screen_cathode.py
git commit -m "feat: add extract_jid and load_and_merge to screen_cathode"
```

---

### Task 3: Filtering and ranking

**Files:**
- Modify: `batterymat/screening_cathode/screen_cathode.py`
- Modify: `tests/test_screen_cathode.py`

- [ ] **Step 1: Add failing tests for filtering and scoring**

Append to `tests/test_screen_cathode.py`:

```python
from batterymat.screening_cathode.screen_cathode import (
    extract_jid,
    load_and_merge,
    filter_cathode_candidates,
    rank_candidates,
    run_screening,
)


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
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 99.9, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 100.1, "ehull": 0.02},
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
    """max_grav_cap == 100.0 should fail (strict >); 100.1 should pass."""
    df = make_merged_df([
        {"jid": "A", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 100.0, "ehull": 0.02},
        {"jid": "B", "avg_voltage": 3.5, "max_voltage": 4.0, "max_grav_cap": 100.1, "ehull": 0.02},
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_screen_cathode.py -k "filter or rank" -v
```
Expected: `ImportError` for `filter_cathode_candidates`, `rank_candidates`

- [ ] **Step 3: Implement `filter_cathode_candidates` and `rank_candidates`**

Append to `batterymat/screening_cathode/screen_cathode.py`:

```python
def filter_cathode_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cathode-specific voltage, capacity, and stability filters.

    Filters (spec Section 3):
      - avg_voltage: 3.0 – 4.5 V
      - max_grav_cap: > 100 mAh/g
      - ehull: <= 0.05 eV
      - max_voltage: <= 5.5 V
    """
    mask = (
        (df["avg_voltage"] >= FILTERS["avg_voltage_min"])
        & (df["avg_voltage"] <= FILTERS["avg_voltage_max"])
        & (df["max_grav_cap"] > FILTERS["max_grav_cap_min"])
        & (df["ehull"] <= FILTERS["ehull_max"])
        & (df["max_voltage"] <= FILTERS["max_voltage_max"])
    )
    return df[mask].reset_index(drop=True)


def _safe_norm(series: pd.Series) -> pd.Series:
    """Min-max normalize; returns 0.5 for all if constant."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite score and sort candidates descending.

    Score = (1/3)*norm(avg_voltage) + (1/3)*norm(max_grav_cap) - (1/3)*norm(ehull)

    If fewer than 2 candidates, returns unranked with a warning.
    """
    if len(df) < 2:
        print(
            "WARNING: Fewer than 2 candidates after filtering — "
            "composite score not computed.",
            file=sys.stdout,
        )
        return df.copy()

    ranked = df.copy()
    ranked["score"] = (
        (1 / 3) * _safe_norm(ranked["avg_voltage"])
        + (1 / 3) * _safe_norm(ranked["max_grav_cap"])
        - (1 / 3) * _safe_norm(ranked["ehull"])
    )
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_screen_cathode.py -k "filter or rank" -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/screen_cathode.py tests/test_screen_cathode.py
git commit -m "feat: add filter_cathode_candidates and rank_candidates"
```

---

### Task 4: `run_screening` entry point and CSV export

**Files:**
- Modify: `batterymat/screening_cathode/screen_cathode.py`
- Modify: `tests/test_screen_cathode.py`

- [ ] **Step 1: Add failing test for run_screening**

Append to `tests/test_screen_cathode.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_screen_cathode.py::test_run_screening_writes_csv -v
```
Expected: `ImportError` for `run_screening`

- [ ] **Step 3: Implement `run_screening`**

Append to `batterymat/screening_cathode/screen_cathode.py`:

```python
# Default paths relative to repo root
_REPO_ROOT = __file__  # batterymat/screening_cathode/screen_cathode.py
import os as _os
_DEFAULT_LI_MIN = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(_REPO_ROOT))),
    "average_voltage", "Li_min.csv",
)
_DEFAULT_OUTPUT = _os.path.join(
    _os.path.dirname(_os.path.abspath(_REPO_ROOT)),
    "cathode_candidates_ranked.csv",
)


def run_screening(
    li_min_df: Optional[pd.DataFrame] = None,
    dft3d_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Full screening pipeline: load → merge → filter → rank → export.

    Args:
        li_min_df: Pre-loaded Li_min.csv DataFrame (or None to read from disk).
        dft3d_df: Pre-loaded JARVIS-DFT DataFrame (or None to fetch from figshare).
        output_path: Where to write the ranked CSV. Defaults to
                     batterymat/screening_cathode/cathode_candidates_ranked.csv.

    Returns:
        Ranked candidates DataFrame.
    """
    if li_min_df is None:
        li_min_df = pd.read_csv(_DEFAULT_LI_MIN)

    if dft3d_df is None:
        try:
            from jarvis.db.figshare import data as jarvis_data
            dft3d_df = pd.DataFrame(jarvis_data("dft_3d"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch JARVIS-DFT dataset: {e}. "
                "Check your internet connection or provide dft3d_df directly."
            ) from e

    merged = load_and_merge(li_min_df, dft3d_df)
    filtered = filter_cathode_candidates(merged)
    ranked = rank_candidates(filtered)

    save_path = output_path or _DEFAULT_OUTPUT
    ranked.to_csv(save_path, index=False)
    print(f"Saved {len(ranked)} cathode candidates to {save_path}")

    return ranked
```

- [ ] **Step 4: Run all screen_cathode tests**

```bash
python -m pytest tests/test_screen_cathode.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/screen_cathode.py tests/test_screen_cathode.py
git commit -m "feat: add run_screening entry point with CSV export"
```

---

## Chunk 2: dft_prep.py

### Task 5: `compute_dft_voltage` helper

**Files:**
- Create: `batterymat/screening_cathode/dft_prep.py`
- Create: `tests/test_dft_prep.py`

- [ ] **Step 1: Write failing tests for voltage helper**

Create `tests/test_dft_prep.py`:

```python
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from batterymat.screening_cathode.dft_prep import compute_dft_voltage


def test_compute_dft_voltage_positive_for_cathode():
    # Cathode: delithiated structure has higher energy than lithiated
    # V = (E_delith - E_lith - n_li * E_li_metal) / n_li
    # With E_delith=10, E_lith=8, n_li=1, E_li=-1.90 => V = (10-8-(-1.90))/1 = 3.90
    v = compute_dft_voltage(
        e_lithiated=-8.0,
        e_delithiated=-10.0 + 2.0,  # = -8.0 + 2.0 energy cost
        n_li=1,
        e_li_metal=-1.90,
    )
    # e_delithiated = -8.0 + 2.0 = -6.0
    # V = (-6.0 - (-8.0) - 1 * (-1.90)) / 1 = (2.0 + 1.90) = 3.90
    assert abs(v - 3.90) < 1e-6


def test_compute_dft_voltage_formula():
    v = compute_dft_voltage(
        e_lithiated=-100.0,
        e_delithiated=-95.0,
        n_li=2,
        e_li_metal=-2.0,
    )
    # V = ((-95) - (-100) - 2*(-2)) / 2 = (5 + 4) / 2 = 4.5
    assert abs(v - 4.5) < 1e-6


def test_compute_dft_voltage_raises_on_zero_n_li():
    with pytest.raises(ValueError, match="n_li"):
        compute_dft_voltage(-100.0, -95.0, 0, -1.90)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dft_prep.py::test_compute_dft_voltage_positive_for_cathode tests/test_dft_prep.py::test_compute_dft_voltage_formula tests/test_dft_prep.py::test_compute_dft_voltage_raises_on_zero_n_li -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `compute_dft_voltage`**

Create `batterymat/screening_cathode/dft_prep.py`:

```python
"""VASP input generation and DFT voltage helper for cathode candidates."""
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Module-level import so tests can patch batterymat.screening_cathode.dft_prep.Vacancy
from jarvis.analysis.defects.vacancy import Vacancy


# ---------------------------------------------------------------------------
# Voltage helper
# ---------------------------------------------------------------------------

def compute_dft_voltage(
    e_lithiated: float,
    e_delithiated: float,
    n_li: int,
    e_li_metal: float,
) -> float:
    """Compute average intercalation voltage from PBE total energies.

    Formula: V = (E_delithiated - E_lithiated - n_li * E_li_metal) / n_li

    Sign convention: for cathodes, E_delithiated > E_lithiated, so V > 0.
    Consistent with screen.py: V = (E_removed - E_full - unary_energy(ion)).

    Args:
        e_lithiated:   Total DFT energy of fully lithiated structure (eV).
        e_delithiated: Total DFT energy of delithiated structure (eV).
        n_li:          Number of Li atoms removed (1 for single-vacancy run).
        e_li_metal:    DFT energy of Li BCC metal per atom (eV/atom).
                       Expected ~-1.90 eV/atom for PBE/Li_sv/ENCUT=520.

    Returns:
        Average voltage in volts (positive for cathodes).
    """
    if n_li == 0:
        raise ValueError("n_li must be > 0")
    return (e_delithiated - e_lithiated - n_li * e_li_metal) / n_li
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dft_prep.py::test_compute_dft_voltage_positive_for_cathode tests/test_dft_prep.py::test_compute_dft_voltage_formula tests/test_dft_prep.py::test_compute_dft_voltage_raises_on_zero_n_li -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/dft_prep.py tests/test_dft_prep.py
git commit -m "feat: add compute_dft_voltage helper"
```

---

### Task 6: VASP input file writers (INCAR, KPOINTS, POTCAR_spec)

**Files:**
- Modify: `batterymat/screening_cathode/dft_prep.py`
- Modify: `tests/test_dft_prep.py`

**Context:** PAW potential labels follow Materials Project recommendations. Elements not in the table fall back to bare element symbol. KPOINTS uses a Gamma-centered 3×3×3 mesh as a starting point (user adjusts for system size). Each TB-mBJ directory also gets a README explaining the CHGCAR dependency.

- [ ] **Step 1: Add failing tests for INCAR, KPOINTS, POTCAR_spec, README**

Append to `tests/test_dft_prep.py`:

```python
from batterymat.screening_cathode.dft_prep import (
    compute_dft_voltage,
    write_pbe_relax_incar,
    write_tmbj_static_incar,
    write_kpoints,
    write_potcar_spec,
    write_tmbj_readme,
)


def test_write_pbe_incar_required_tags(tmp_path):
    write_pbe_relax_incar(tmp_path)
    incar = (tmp_path / "INCAR").read_text()
    for tag in ["IBRION = 2", "NSW = 100", "ISIF = 3", "ENCUT = 520",
                "EDIFF = 1E-6", "EDIFFG = -0.01", "LCHARG = .TRUE."]:
        assert tag in incar, f"Missing tag: {tag}"


def test_write_tmbj_incar_required_tags(tmp_path):
    write_tmbj_static_incar(tmp_path)
    incar = (tmp_path / "INCAR").read_text()
    for tag in ["IBRION = -1", "NSW = 0", "METAGGA = MBJ",
                "LASPH = .TRUE.", "ICHARG = 11", "LORBIT = 11",
                "NEDOS = 2000"]:
        assert tag in incar, f"Missing tag: {tag}"


def test_write_kpoints_creates_file(tmp_path):
    write_kpoints(tmp_path)
    kpoints = (tmp_path / "KPOINTS").read_text()
    assert "Gamma" in kpoints or "gamma" in kpoints.lower()


def test_write_potcar_spec_known_element(tmp_path):
    write_potcar_spec(tmp_path, ["Li", "Mn", "O"])
    spec = (tmp_path / "POTCAR_spec").read_text()
    assert "Li_sv" in spec
    assert "Mn_pv" in spec
    assert "O" in spec


def test_write_potcar_spec_unknown_element_fallback(tmp_path):
    write_potcar_spec(tmp_path, ["Li", "Xx"])  # Xx is not in the table
    spec = (tmp_path / "POTCAR_spec").read_text()
    assert "Xx" in spec  # bare symbol used as fallback


def test_write_tmbj_readme_mentions_chgcar(tmp_path):
    pbe_dir = tmp_path / "pbe_run"
    pbe_dir.mkdir()
    write_tmbj_readme(tmp_path, pbe_dir)
    readme = (tmp_path / "README").read_text()
    assert "CHGCAR" in readme
    assert str(pbe_dir) in readme
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dft_prep.py -k "write" -v
```
Expected: `ImportError` for new functions

- [ ] **Step 3: Implement the VASP input file writers**

Append to `batterymat/screening_cathode/dft_prep.py`:

```python
# ---------------------------------------------------------------------------
# Materials Project recommended PAW potentials (subset)
# Ref: https://materialsproject.org/docs/methodology/pseudopotentials
# ---------------------------------------------------------------------------
_MP_PAW = {
    "Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv",
    "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv",
    "Sc": "Sc_sv", "Ti": "Ti_pv", "V": "V_pv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co", "Ni": "Ni_pv",
    "Cu": "Cu_pv", "Zn": "Zn", "Ga": "Ga_d", "Ge": "Ge_d",
    "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd", "Ag": "Ag",
    "Sn": "Sn_d", "Sb": "Sb",
    "W": "W_pv", "Re": "Re_pv", "Os": "Os_pv", "Ir": "Ir",
    "Pt": "Pt", "Au": "Au",
    "O": "O", "S": "S", "Se": "Se", "Te": "Te",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "N": "N", "P": "P", "As": "As",
    "C": "C", "Si": "Si", "B": "B",
    "Al": "Al", "In": "In_d",
    "La": "La", "Ce": "Ce", "Pr": "Pr_3", "Nd": "Nd_3",
    "Sm": "Sm_3", "Eu": "Eu_2", "Gd": "Gd_3", "Tb": "Tb_3",
    "Dy": "Dy_3", "Ho": "Ho_3", "Er": "Er_3", "Tm": "Tm_3",
    "Yb": "Yb_2", "Lu": "Lu_3",
}

_PBE_INCAR = """\
IBRION = 2
NSW = 100
ISIF = 3
ENCUT = 520
EDIFF = 1E-6
EDIFFG = -0.01
PREC = Accurate
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
LCHARG = .TRUE.
"""

_TMBJ_INCAR = """\
IBRION = -1
NSW = 0
ENCUT = 520
EDIFF = 1E-6
METAGGA = MBJ
LASPH = .TRUE.
ICHARG = 11
PREC = Accurate
ISMEAR = 0
SIGMA = 0.05
LORBIT = 11
NEDOS = 2000
LWAVE = .FALSE.
LCHARG = .FALSE.
"""

_KPOINTS = """\
Automatic mesh
0
Gamma
3 3 3
0 0 0
"""


def write_pbe_relax_incar(directory) -> None:
    """Write PBE relaxation INCAR to directory."""
    Path(directory, "INCAR").write_text(_PBE_INCAR)


def write_tmbj_static_incar(directory) -> None:
    """Write TB-mBJ static INCAR to directory."""
    Path(directory, "INCAR").write_text(_TMBJ_INCAR)


def write_kpoints(directory) -> None:
    """Write a Gamma-centered 3x3x3 KPOINTS to directory."""
    Path(directory, "KPOINTS").write_text(_KPOINTS)


def write_potcar_spec(directory, elements: List[str]) -> None:
    """Write POTCAR_spec listing recommended PAW labels for elements.

    Unknown elements fall back to the bare element symbol.
    """
    labels = [_MP_PAW.get(el, el) for el in elements]
    Path(directory, "POTCAR_spec").write_text("\n".join(labels) + "\n")


def write_tmbj_readme(tmbj_dir, pbe_dir) -> None:
    """Write README in a TB-mBJ directory explaining CHGCAR dependency."""
    content = f"""\
TB-mBJ Static Run — Pre-submission checklist
=============================================

Before submitting this calculation:

1. Copy CHGCAR from the PBE relaxation:
       cp {pbe_dir}/CHGCAR {tmbj_dir}/CHGCAR

2. Replace POSCAR with the relaxed geometry:
       cp {pbe_dir}/CONTCAR {tmbj_dir}/POSCAR

3. The INCAR already includes ICHARG = 11 to read the CHGCAR.

Reference: Li BCC metal energy for voltage calculation ~ -1.90 eV/atom
(PBE, Li_sv PAW, ENCUT=520, consistent with Materials Project).
"""
    Path(tmbj_dir, "README").write_text(content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dft_prep.py -k "write" -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/dft_prep.py tests/test_dft_prep.py
git commit -m "feat: add VASP input file writers (INCAR, KPOINTS, POTCAR_spec, README)"
```

---

### Task 7: Delithiated structure generation via ALIGNN + Vacancy

**Files:**
- Modify: `batterymat/screening_cathode/dft_prep.py`
- Modify: `tests/test_dft_prep.py`

**Context:** `Vacancy.generate_defects()` takes an `Atoms` object and returns a list of `Vacancy` objects. Each has a `_defect_structure` attribute which is an `Atoms` object. We score each defect structure with ALIGNN and pick the lowest energy one. If `n_Li == 1`, we skip vacancy enumeration and just remove the single Li directly.

- [ ] **Step 1: Add failing tests for delithiated structure generation**

Append to `tests/test_dft_prep.py`:

```python
from batterymat.screening_cathode.dft_prep import get_delithiated_structure


def _make_mock_atoms(elements):
    """Create a minimal mock Atoms-like object."""
    mock = MagicMock()
    mock.elements = elements
    mock.num_atoms = len(elements)
    return mock


def test_get_delithiated_raises_on_no_li():
    atoms = _make_mock_atoms(["Mn", "O", "O"])
    with pytest.raises(ValueError, match="no Li atoms"):
        get_delithiated_structure(atoms)


def test_get_delithiated_single_li_skips_vacancy_enumeration():
    """When n_Li == 1, return the host structure directly (no Vacancy call)."""
    # Build a real minimal Atoms via jarvis if available, else skip
    pytest.importorskip("jarvis")
    from jarvis.core.atoms import Atoms

    # Simple LiMnO2-like structure: 1 Li + host
    atoms = Atoms(
        lattice_mat=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        elements=["Li", "Mn", "O", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cartesian=False,
    )
    with patch("batterymat.screening_cathode.dft_prep.Vacancy") as mock_vac:
        result = get_delithiated_structure(atoms)
        mock_vac.assert_not_called()
    assert "Li" not in result.elements


def test_get_delithiated_multi_li_uses_alignn_ranking():
    """When n_Li > 1, calls Vacancy.generate_defects and ranks by ALIGNN energy."""
    pytest.importorskip("jarvis")
    from jarvis.core.atoms import Atoms

    atoms = Atoms(
        lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        elements=["Li", "Li", "Mn", "O", "O", "O", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
        ],
        cartesian=False,
    )

    # Mock Vacancy to return two fake defect structures
    mock_defect_1 = MagicMock()
    mock_defect_1._defect_structure = MagicMock()
    mock_defect_2 = MagicMock()
    mock_defect_2._defect_structure = MagicMock()

    # Mock ALIGNN to return energies: defect_1 = -10.0, defect_2 = -12.0
    # Lower energy → defect_2 should be selected
    def mock_energy(defect_atoms):
        if defect_atoms is mock_defect_1._defect_structure:
            return -10.0
        return -12.0

    with patch("batterymat.screening_cathode.dft_prep.Vacancy") as MockVac, \
         patch("batterymat.screening_cathode.dft_prep._alignn_energy", side_effect=mock_energy):
        instance = MockVac.return_value
        instance.generate_defects.return_value = [mock_defect_1, mock_defect_2]
        result = get_delithiated_structure(atoms)

    assert result is mock_defect_2._defect_structure
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dft_prep.py -k "delithiated" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `get_delithiated_structure` and `_alignn_energy`**

Append to `batterymat/screening_cathode/dft_prep.py`:

```python
# ---------------------------------------------------------------------------
# Delithiated structure generation
# ---------------------------------------------------------------------------

def _alignn_energy(atoms) -> float:
    """Compute ALIGNN ML force field energy for a jarvis Atoms object."""
    try:
        from alignn.ff.ff import AlignnAtomwiseCalculator, wt01_path
    except ImportError as e:
        raise ImportError(
            f"ALIGNN not available: {e}. Verify your ALIGNN installation."
        ) from e
    calc = AlignnAtomwiseCalculator(path=wt01_path(), stress_wt=0.3)
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calc
    return ase_atoms.get_potential_energy()


def get_delithiated_structure(atoms, model_path=None):
    """Return the lowest-energy single-Li-vacancy structure.

    Uses ALIGNN-ranked Wyckoff-inequivalent vacancy configurations.
    If n_Li == 1, returns the fully delithiated host directly.
    If n_Li == 0, raises ValueError.

    Args:
        atoms: jarvis.core.atoms.Atoms object (fully lithiated).
        model_path: Unused; ALIGNN wt01_path() is always used (for API consistency).

    Returns:
        jarvis.core.atoms.Atoms — delithiated structure.
    """
    li_indices = [i for i, el in enumerate(atoms.elements) if el == "Li"]
    n_li = len(li_indices)

    if n_li == 0:
        raise ValueError(
            f"Structure has no Li atoms — cannot generate delithiated structure."
        )

    if n_li == 1:
        # Single Li: remove it directly, no vacancy enumeration needed
        return atoms.remove_site_by_index(li_indices[0])

    # Multiple Li sites: enumerate Wyckoff-inequivalent vacancies and rank by energy
    vac = Vacancy(atoms=atoms)
    defects = vac.generate_defects(enforce_c_size=None, extend=0)
    li_defects = [d for d in defects if d._symbol == "Li"]

    if not li_defects:
        # Fallback: remove the first Li
        warnings.warn(
            "Vacancy enumeration returned no Li defects; removing first Li atom."
        )
        return atoms.remove_site_by_index(li_indices[0])

    energies = [_alignn_energy(d._defect_structure) for d in li_defects]
    best = li_defects[energies.index(min(energies))]
    return best._defect_structure
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dft_prep.py -k "delithiated" -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/dft_prep.py tests/test_dft_prep.py
git commit -m "feat: add get_delithiated_structure with ALIGNN-ranked vacancy selection"
```

---

### Task 8: `generate_inputs` — full VASP directory generation

**Files:**
- Modify: `batterymat/screening_cathode/dft_prep.py`
- Modify: `tests/test_dft_prep.py`

**Context:** For each JID, create 4 subdirectories under `output_dir/JVASP-XXXXX/`. The `atoms` column in the JARVIS-DFT DataFrame contains a dict that must be deserialized with `Atoms.from_dict()`.

- [ ] **Step 1: Add failing tests for `generate_inputs`**

Append to `tests/test_dft_prep.py`:

```python
from batterymat.screening_cathode.dft_prep import generate_inputs


def _make_mock_dft3d(jid, elements):
    """Return a minimal fake dft3d DataFrame row."""
    import pandas as pd
    from jarvis.core.atoms import Atoms
    atoms = Atoms(
        lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        elements=elements,
        coords=[[float(i) / len(elements)] * 3 for i in range(len(elements))],
        cartesian=False,
    )
    return pd.DataFrame([{"jid": jid, "atoms": atoms.to_dict()}])


def test_generate_inputs_creates_four_dirs(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-999", ["Li", "Mn", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["Mn", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        created = generate_inputs(["JVASP-999"], output_dir=str(tmp_path), dft3d_df=dft3d)

    jid_dir = tmp_path / "JVASP-999"
    assert jid_dir.exists()
    for sub in ["1_pbe_relax_lithiated", "2_pbe_relax_delithiated",
                "3_tmbj_static_lithiated", "4_tmbj_static_delithiated"]:
        assert (jid_dir / sub).exists(), f"Missing: {sub}"

    assert len(created) == 1
    assert str(jid_dir) in created[0] or created[0] == str(jid_dir)


def test_generate_inputs_each_dir_has_required_files(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-888", ["Li", "Co", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["Co", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        generate_inputs(["JVASP-888"], output_dir=str(tmp_path), dft3d_df=dft3d)

    jid_dir = tmp_path / "JVASP-888"
    for sub in ["1_pbe_relax_lithiated", "2_pbe_relax_delithiated"]:
        for f in ["POSCAR", "INCAR", "KPOINTS", "POTCAR_spec"]:
            assert (jid_dir / sub / f).exists(), f"Missing {sub}/{f}"
    for sub in ["3_tmbj_static_lithiated", "4_tmbj_static_delithiated"]:
        for f in ["POSCAR", "INCAR", "KPOINTS", "POTCAR_spec", "README"]:
            assert (jid_dir / sub / f).exists(), f"Missing {sub}/{f}"


def test_generate_inputs_skips_zero_li(tmp_path, capsys):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-777", ["Mn", "O", "O"])  # No Li
    created = generate_inputs(["JVASP-777"], output_dir=str(tmp_path), dft3d_df=dft3d)
    captured = capsys.readouterr()
    assert len(created) == 0
    assert "WARNING" in captured.out
    assert "JVASP-777" in captured.out


def test_generate_inputs_returns_list_of_paths(tmp_path):
    pytest.importorskip("jarvis")
    dft3d = _make_mock_dft3d("JVASP-555", ["Li", "V", "O", "O", "O", "O", "O"])
    with patch("batterymat.screening_cathode.dft_prep.get_delithiated_structure") as mock_delith:
        from jarvis.core.atoms import Atoms
        mock_delith.return_value = Atoms(
            lattice_mat=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            elements=["V", "O", "O"],
            coords=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cartesian=False,
        )
        result = generate_inputs(["JVASP-555"], output_dir=str(tmp_path), dft3d_df=dft3d)
    assert isinstance(result, list)
    assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dft_prep.py -k "generate_inputs" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `generate_inputs`**

Append to `batterymat/screening_cathode/dft_prep.py`:

```python
# ---------------------------------------------------------------------------
# POSCAR writer
# ---------------------------------------------------------------------------

def _write_poscar(directory: Path, atoms) -> None:
    """Write VASP POSCAR from a jarvis Atoms object."""
    from jarvis.io.vasp.inputs import Poscar
    poscar = Poscar(atoms)
    poscar.write_file(str(Path(directory, "POSCAR")))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "dft_inputs"


def generate_inputs(
    jids: List[str],
    output_dir: Optional[str] = None,
    dft3d_df=None,
) -> List[str]:
    """Generate 4 VASP input directories per JID for DFT validation.

    Directory layout per JID:
        <output_dir>/<JID>/
            1_pbe_relax_lithiated/   POSCAR INCAR KPOINTS POTCAR_spec
            2_pbe_relax_delithiated/ POSCAR INCAR KPOINTS POTCAR_spec
            3_tmbj_static_lithiated/ POSCAR INCAR KPOINTS POTCAR_spec README
            4_tmbj_static_delithiated/ POSCAR INCAR KPOINTS POTCAR_spec README

    Args:
        jids:       List of JARVIS JIDs to process.
        output_dir: Root directory for output. Defaults to
                    batterymat/screening_cathode/dft_inputs/.
        dft3d_df:   Pre-loaded JARVIS-DFT DataFrame. Fetched from figshare if None.

    Returns:
        List of JID-level directory paths that were successfully created.
    """
    from jarvis.core.atoms import Atoms as JAtoms

    root = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)

    if dft3d_df is None:
        try:
            from jarvis.db.figshare import data as jarvis_data
            import pandas as pd
            dft3d_df = pd.DataFrame(jarvis_data("dft_3d"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch JARVIS-DFT dataset: {e}. "
                "Provide dft3d_df directly."
            ) from e

    jid_to_row = {row["jid"]: row for _, row in dft3d_df.iterrows()}
    created = []

    for jid in jids:
        if jid not in jid_to_row:
            print(f"WARNING: {jid} not found in JARVIS-DFT dataset — skipping.", file=sys.stdout)
            continue

        row = jid_to_row[jid]
        lith_atoms = JAtoms.from_dict(row["atoms"])
        unique_elements = list(dict.fromkeys(lith_atoms.elements))

        # Check for Li
        if "Li" not in lith_atoms.elements:
            print(f"WARNING: {jid} has no Li atoms — skipping.", file=sys.stdout)
            continue

        # Generate delithiated structure
        try:
            delith_atoms = get_delithiated_structure(lith_atoms)
        except Exception as e:
            print(f"WARNING: Failed to generate delithiated structure for {jid}: {e} — skipping.", file=sys.stdout)
            continue

        delith_elements = list(dict.fromkeys(delith_atoms.elements))
        jid_dir = root / jid
        jid_dir.mkdir(parents=True, exist_ok=True)

        # --- Run 1: PBE relax, lithiated ---
        run1 = jid_dir / "1_pbe_relax_lithiated"
        run1.mkdir(exist_ok=True)
        _write_poscar(run1, lith_atoms)
        write_pbe_relax_incar(run1)
        write_kpoints(run1)
        write_potcar_spec(run1, unique_elements)

        # --- Run 2: PBE relax, delithiated ---
        run2 = jid_dir / "2_pbe_relax_delithiated"
        run2.mkdir(exist_ok=True)
        _write_poscar(run2, delith_atoms)
        write_pbe_relax_incar(run2)
        write_kpoints(run2)
        write_potcar_spec(run2, delith_elements)

        # --- Run 3: TB-mBJ static, lithiated ---
        run3 = jid_dir / "3_tmbj_static_lithiated"
        run3.mkdir(exist_ok=True)
        _write_poscar(run3, lith_atoms)  # user replaces with CONTCAR from run1
        write_tmbj_static_incar(run3)
        write_kpoints(run3)
        write_potcar_spec(run3, unique_elements)
        write_tmbj_readme(run3, run1)

        # --- Run 4: TB-mBJ static, delithiated ---
        run4 = jid_dir / "4_tmbj_static_delithiated"
        run4.mkdir(exist_ok=True)
        _write_poscar(run4, delith_atoms)  # user replaces with CONTCAR from run2
        write_tmbj_static_incar(run4)
        write_kpoints(run4)
        write_potcar_spec(run4, delith_elements)
        write_tmbj_readme(run4, run2)

        print(f"Generated DFT inputs for {jid} in {jid_dir}")
        created.append(str(jid_dir))

    return created
```

- [ ] **Step 4: Run all dft_prep tests**

```bash
python -m pytest tests/test_dft_prep.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add batterymat/screening_cathode/dft_prep.py tests/test_dft_prep.py
git commit -m "feat: add generate_inputs for VASP directory creation"
```

---

## Chunk 3: Notebook

### Task 9: `cathode_filtering.ipynb`

**Files:**
- Create: `batterymat/screening_cathode/cathode_filtering.ipynb`

**Context:** The notebook is the interactive user-facing layer. It does not contain logic — it calls `screen_cathode.run_screening()` and `dft_prep.generate_inputs()`. Created programmatically with `nbformat` to ensure valid JSON structure.

- [ ] **Step 1: Create the notebook**

```python
# Run this script once to generate the notebook
# python create_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Cathode Candidate Screening\n\nFilters Li-ion cathode candidates from ML-screened data and generates VASP DFT inputs for top candidates."),

    nbf.v4.new_markdown_cell("## 1. Load & Filter"),
    nbf.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.abspath('../..'))  # repo root

import pandas as pd
from jarvis.db.figshare import data as jarvis_data
from batterymat.screening_cathode.screen_cathode import run_screening

# Load data (fetches JARVIS-DFT on first run, ~2 min)
dft3d_df = pd.DataFrame(jarvis_data('dft_3d'))
li_min_df = pd.read_csv('../screening_anode/Li_min.csv')

candidates = run_screening(li_min_df=li_min_df, dft3d_df=dft3d_df)
print(f"Found {len(candidates)} candidates after filtering")
candidates.head(20)
"""),

    nbf.v4.new_markdown_cell("## 2. Visualize"),
    nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt
%matplotlib inline

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

candidates['avg_voltage'].hist(ax=axes[0], bins=30, color='steelblue')
axes[0].set_xlabel('Average Voltage (V)')
axes[0].set_ylabel('Count')
axes[0].set_title('Average Voltage Distribution')

candidates['max_grav_cap'].hist(ax=axes[1], bins=30, color='darkorange')
axes[1].set_xlabel('Max Gravimetric Capacity (mAh/g)')
axes[1].set_title('Capacity Distribution')

candidates['ehull'].hist(ax=axes[2], bins=30, color='seagreen')
axes[2].set_xlabel('E-hull (eV)')
axes[2].set_title('Stability Distribution')

plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_code_cell("""\
# Scatter: voltage vs capacity, colored by composite score
sc = plt.scatter(
    candidates['avg_voltage'],
    candidates['max_grav_cap'],
    c=candidates['score'],
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(sc, label='Composite Score')
plt.xlabel('Average Voltage (V)')
plt.ylabel('Max Gravimetric Capacity (mAh/g)')
plt.title('Cathode Candidates: Voltage vs Capacity')
plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("## 3. Select Candidates & Generate DFT Inputs"),
    nbf.v4.new_code_cell("""\
# Review top candidates
candidates[['jid', 'formula', 'avg_voltage', 'max_grav_cap', 'ehull', 'optb88vdw_bandgap', 'score']].head(30)
"""),
    nbf.v4.new_code_cell("""\
# Set your selected JIDs here
selected_jids = [
    # "JVASP-XXXXX",
    # "JVASP-YYYYY",
]

from batterymat.screening_cathode.dft_prep import generate_inputs

created_dirs = generate_inputs(
    jids=selected_jids,
    dft3d_df=dft3d_df,
    # output_dir="/path/to/custom/output"  # optional override
)

print(f"Created {len(created_dirs)} DFT input sets:")
for d in created_dirs:
    print(f"  {d}")
"""),
]

with open('batterymat/screening_cathode/cathode_filtering.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created.")
```

Run from repo root:

```bash
python -c "
import sys; sys.path.insert(0, '.')
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell('# Cathode Candidate Screening\n\nFilters Li-ion cathode candidates from ML-screened data and generates VASP DFT inputs for top candidates.'),
    nbf.v4.new_markdown_cell('## 1. Load & Filter'),
    nbf.v4.new_code_cell('import sys, os\nsys.path.insert(0, os.path.abspath(\"../..\"))\n\nimport pandas as pd\nfrom jarvis.db.figshare import data as jarvis_data\nfrom batterymat.screening_cathode.screen_cathode import run_screening\n\ndft3d_df = pd.DataFrame(jarvis_data(\"dft_3d\"))\nli_min_df = pd.read_csv(\"../average_voltage/Li_min.csv\")\n\ncandidates = run_screening(li_min_df=li_min_df, dft3d_df=dft3d_df)\nprint(f\"Found {len(candidates)} candidates after filtering\")\ncandidates.head(20)'),
    nbf.v4.new_markdown_cell('## 2. Visualize'),
    nbf.v4.new_code_cell('import matplotlib.pyplot as plt\n%matplotlib inline\n\nfig, axes = plt.subplots(1, 3, figsize=(14, 4))\ncandidates[\"avg_voltage\"].hist(ax=axes[0], bins=30, color=\"steelblue\")\naxes[0].set_xlabel(\"Average Voltage (V)\")\naxes[0].set_ylabel(\"Count\")\naxes[0].set_title(\"Average Voltage Distribution\")\ncandidates[\"max_grav_cap\"].hist(ax=axes[1], bins=30, color=\"darkorange\")\naxes[1].set_xlabel(\"Max Gravimetric Capacity (mAh/g)\")\naxes[1].set_title(\"Capacity Distribution\")\ncandidates[\"ehull\"].hist(ax=axes[2], bins=30, color=\"seagreen\")\naxes[2].set_xlabel(\"E-hull (eV)\")\naxes[2].set_title(\"Stability Distribution\")\nplt.tight_layout()\nplt.show()'),
    nbf.v4.new_code_cell('sc = plt.scatter(candidates[\"avg_voltage\"], candidates[\"max_grav_cap\"], c=candidates[\"score\"], cmap=\"viridis\", alpha=0.7)\nplt.colorbar(sc, label=\"Composite Score\")\nplt.xlabel(\"Average Voltage (V)\")\nplt.ylabel(\"Max Gravimetric Capacity (mAh/g)\")\nplt.title(\"Cathode Candidates: Voltage vs Capacity\")\nplt.tight_layout()\nplt.show()'),
    nbf.v4.new_markdown_cell('## 3. Select Candidates & Generate DFT Inputs'),
    nbf.v4.new_code_cell('candidates[[\"jid\", \"formula\", \"avg_voltage\", \"max_grav_cap\", \"ehull\", \"optb88vdw_bandgap\", \"score\"]].head(30)'),
    nbf.v4.new_code_cell('selected_jids = [\n    # \"JVASP-XXXXX\",\n]\n\nfrom batterymat.screening_cathode.dft_prep import generate_inputs\n\ncreated_dirs = generate_inputs(\n    jids=selected_jids,\n    dft3d_df=dft3d_df,\n)\n\nprint(f\"Created {len(created_dirs)} DFT input sets:\")\nfor d in created_dirs:\n    print(f\"  {d}\")'),
]

with open('batterymat/screening_cathode/cathode_filtering.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook created.')
"
```

- [ ] **Step 2: Verify the notebook is valid JSON**

```bash
python -c "import json; json.load(open('batterymat/screening_cathode/cathode_filtering.ipynb')); print('valid JSON')"
```
Expected: `valid JSON`

- [ ] **Step 3: Verify the notebook opens in Jupyter (manual check)**

```bash
jupyter nbconvert --to script batterymat/screening_cathode/cathode_filtering.ipynb --stdout 2>/dev/null | head -20
```
Expected: Python source lines printed (confirms nbformat is parseable)

- [ ] **Step 4: Commit**

```bash
git add batterymat/screening_cathode/cathode_filtering.ipynb
git commit -m "feat: add cathode_filtering.ipynb for interactive screening and DFT export"
```

---

### Task 10: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASS

- [ ] **Step 2: Smoke-test screen_cathode on real data (optional, requires network)**

```bash
python -c "
from batterymat.screening_cathode.screen_cathode import run_screening
df = run_screening()
print(f'Candidates: {len(df)}')
print(df[['jid','formula','avg_voltage','max_grav_cap','ehull','score']].head(5).to_string())
"
```
Expected: prints a table of top cathode candidates

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete cathode screening + DFT prep pipeline"
```
