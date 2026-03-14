# Cathode Screening + DFT Validation Pipeline — Design Spec

**Date:** 2026-03-11
**Project:** BatteryMat (NIST)
**Scope:** Li-ion cathode candidate screening from existing ML-screened data, followed by automated VASP input generation for DFT validation.

---

## 1. Overview

This pipeline adds a dedicated cathode screening and DFT preparation stage on top of the existing anode screening infrastructure. It operates in two independent stages:

- **Stage 1 — Screening:** Filter and rank Li-ion cathode candidates from the existing `Li_min.csv` dataset using cathode-specific electrochemical thresholds and a composite performance score.
- **Stage 2 — DFT Prep:** For user-selected candidates, generate complete VASP input directories for geometry optimization (PBE) and electronic structure (TB-mBJ) calculations.

No re-running of ALIGNN force field inference is required for Stage 1. ALIGNN is used in Stage 2 only to rank symmetry-inequivalent Li-vacancy configurations.

---

## 2. Directory Structure

```
batterymat/screening_cathode/
├── screen_cathode.py           # Filtering, ranking, CSV export
├── dft_prep.py                 # VASP input directory generation
└── cathode_filtering.ipynb     # Interactive exploration and candidate selection

docs/superpowers/specs/
└── 2026-03-11-cathode-screening-dft-design.md  # This document
```

---

## 3. Stage 1: Screening (`screen_cathode.py`)

### Input
- `batterymat/average_voltage/Li_min.csv` — existing ML-screened Li structures (~7,474 entries, filtered by `Battery_Filtering.ipynb`) with columns: `name`, `avg_voltage`, `max_voltage`, `max_grav_cap`, `max_vol_cap`, `voltage_profile`, `grav_capacity_profile`, `vol_capacity_profile`
- JARVIS-DFT 3D dataset (fetched via `jarvis.db.figshare.data('dft_3d')`) — provides `ehull`, `optb88vdw_bandgap`, `formula`, `atoms`

The two datasets are joined on the `jid` column (inner join). JIDs present in `Li_min.csv` but absent from the JARVIS-DFT dataset are dropped with a warning logged to stdout listing the missing JIDs and count. If the JARVIS-DFT network fetch fails entirely (e.g., offline HPC environment), the script raises a `RuntimeError` with a message directing the user to download the dataset manually via `jarvis.db.figshare` or use a cached local copy. `optb88vdw_bandgap` from JARVIS-DFT is included in the output CSV for reference but is not used in filtering or scoring.

### Filters

| Property | Threshold | Rationale |
|----------|-----------|-----------|
| `avg_voltage` | 3.0 – 4.5 V | Cathode operating range (cf. NMC ~3.8 V, LFP ~3.4 V) |
| `max_grav_cap` | > 100 mAh/g | Minimum practical gravimetric capacity |
| `ehull` | ≤ 0.05 eV | Thermodynamic stability (existing pipeline threshold) |
| `max_voltage` | ≤ 5.5 V | Guards against unphysical predictions; carried forward from the existing `average_voltage/` pipeline for consistency. Note: this is redundant for candidates already satisfying `avg_voltage ≤ 4.5 V`, but is retained to catch edge cases where the voltage profile is highly non-uniform (e.g., avg ≤ 4.5 V but a single spike reaches 5.5+ V). |
| `min_voltage` | already filtered (> 1 V) | `Li_min.csv` is the output of `Battery_Filtering.ipynb` which already applied `min_voltage > 1 V`. This filter is not re-applied here but is documented for traceability. |

### Composite Score

Candidates are ranked by a normalized composite score:

```
score = (1/3) * norm(avg_voltage) + (1/3) * norm(max_grav_cap) - (1/3) * norm(ehull)
```

Where `norm(x) = (x - min(x)) / (max(x) - min(x))` across the filtered candidate set. Equal weights (1/3 each, summing to 1.0) are used by default. If all candidates share the same value for a property (i.e., `max(x) == min(x)`), `norm(x)` returns 0.5 for all candidates for that property. If fewer than 2 candidates pass all filters, scoring is skipped and the candidates are returned unranked with a warning.

### Output
- `batterymat/screening_cathode/cathode_candidates_ranked.csv` — filtered and ranked candidates with all screening metrics and composite score. Columns include all filter properties plus `score`, `optb88vdw_bandgap`, `formula`, `max_vol_cap`, and `vol_capacity_profile` (pass-through for reference; not used in filtering or scoring).

---

## 4. Stage 2: DFT Prep (`dft_prep.py`)

### Input
- A list of JARVIS IDs (JIDs) selected by the user from the ranked candidate list
- JARVIS-DFT 3D dataset (for structure retrieval)

### Delithiated Structure Generation

For each candidate:
1. Retrieve fully lithiated `Atoms` object from JARVIS-DFT by JID
2. Use `jarvis.analysis.defects.vacancy.Vacancy.generate_defects()` to enumerate symmetry-inequivalent single-Li-vacancy configurations (Wyckoff-based)
3. Score each vacancy structure using the ALIGNN ML force field. The calculator is initialized with `wt01_path()` (the default ALIGNN-FF weights, resolved automatically by `from alignn.ff.ff import wt01_path`), matching the model used in `screening_anode/screen.py`. No user-provided path is needed unless overriding the default weights. This import path is used in the existing codebase and is tested against the installed ALIGNN version; if the import fails, a clear `ImportError` message directs the user to verify their ALIGNN installation.
4. Select the lowest-energy configuration as the delithiated structure for DFT

**Single-Li-vacancy scope:** For materials with multiple Li sites (e.g., layered oxides), the spec addresses single-Li removal only (x = 1 - 1/n_Li delithiation). Full delithiation (all Li removed) is handled as a separate option: when `n_Li == 1` in the unit cell, the fully delithiated host is used directly. Intermediate partial delithiation states (0 < x < 1) are out of scope for this iteration and noted as a future extension. If a JID resolves to a structure with `n_Li == 0` (no Li atoms found in the JARVIS `atoms` object), that JID is skipped with a warning and excluded from DFT input generation.

### VASP Input Directory Layout

For each selected JID, four input directories are generated. The output root defaults to `batterymat/screening_cathode/dft_inputs/` (relative to the repo root) and can be overridden via the `output_dir` parameter of `generate_inputs`.

```
batterymat/screening_cathode/dft_inputs/   (default output root)
└── JVASP-XXXXX/
    ├── 1_pbe_relax_lithiated/
    │   ├── POSCAR
    │   ├── INCAR
    │   ├── KPOINTS
    │   └── POTCAR_spec
    ├── 2_pbe_relax_delithiated/
    │   ├── POSCAR
    │   ├── INCAR
    │   ├── KPOINTS
    │   └── POTCAR_spec
    ├── 3_tmbj_static_lithiated/
    │   ├── POSCAR        (copy from 1_pbe_relax_lithiated — to be replaced with relaxed CONTCAR)
    │   ├── INCAR
    │   ├── KPOINTS
    │   └── POTCAR_spec
    └── 4_tmbj_static_delithiated/
        ├── POSCAR        (copy from 2_pbe_relax_delithiated — to be replaced with relaxed CONTCAR)
        ├── INCAR
        ├── KPOINTS
        └── POTCAR_spec
```

> **Note:** POSCAR files for TB-mBJ static runs (3 and 4) are initially seeded with the unrelaxed structure. The user must replace them with the `CONTCAR` from the corresponding PBE relaxation before running.

`POTCAR_spec` is a plain-text file listing recommended PAW potential labels (e.g., `Li_sv`, `Mn_pv`). Potential labels follow the **Materials Project / VASP recommended PAW settings** (ref: https://materialsproject.org/docs/methodology/pseudopotentials). Actual POTCAR files are not generated as they require a licensed VASP installation.

### INCAR Settings

**PBE Relaxation (runs 1 and 2):**

```
IBRION = 2       # Conjugate gradient ionic relaxation
NSW    = 100     # Max ionic steps
ISIF   = 3       # Relax ions + cell shape + cell volume
ENCUT  = 520     # Plane-wave cutoff (eV)
EDIFF  = 1E-6    # Electronic convergence
EDIFFG = -0.01   # Ionic convergence (eV/Å)
PREC   = Accurate
ISMEAR = 0       # Gaussian smearing
SIGMA  = 0.05
LWAVE  = .TRUE.  # Write WAVECAR (optional, for restarting)
LCHARG = .TRUE.  # Write CHGCAR — required as input to TB-mBJ static (runs 3/4)
```

**TB-mBJ Static (runs 3 and 4):**

```
IBRION  = -1          # No ionic relaxation
NSW     = 0
ENCUT   = 520
EDIFF   = 1E-6
METAGGA = MBJ         # Tran-Blaha modified Becke-Johnson
LASPH   = .TRUE.      # Required for meta-GGA
ICHARG  = 11          # Read CHGCAR from PBE run (required for TB-mBJ)
PREC    = Accurate
ISMEAR  = 0
SIGMA   = 0.05
LORBIT  = 11          # Site-projected DOS
NEDOS   = 2000        # DOS energy grid points
LWAVE   = .FALSE.
LCHARG  = .FALSE.
```

**TB-mBJ workflow dependency:** TB-mBJ requires an initial charge density from a preceding GGA run. The generated TB-mBJ INCAR will include `ICHARG = 11` by default. The user must copy the `CHGCAR` from the corresponding PBE relaxation (run 1 → run 3, run 2 → run 4) into the TB-mBJ directory before submitting. A `README` file is generated in each TB-mBJ directory listing the exact file to copy. The POSCAR must also be replaced with the relaxed `CONTCAR` as noted above.

### Voltage Calculation (post-DFT)

After DFT runs complete, voltage is computed from PBE total energies:

```
V = (E_PBE_delithiated - E_PBE_lithiated - n_Li * E_Li_metal) / n_Li
```

**Sign convention:** For cathodes, delithiation raises energy (E_delithiated > E_lithiated), so V > 0. This is consistent with the existing `screen.py` formula `V = (E_removed - E_full - unary_energy(ion))` — "removed" corresponds to "delithiated" and "full" corresponds to "lithiated." The formulas are equivalent; the sign convention is the same.

**`n_Li` definition:** In the voltage formula, `n_Li` is the number of Li atoms removed (1 for a single-vacancy run). This is not the total number of Li atoms in the unit cell.

`E_Li_metal` is the DFT energy of Li BCC metal per atom, computed with the **same VASP settings** (ENCUT=520, PBE, `Li_sv` PAW potential) as the cathode relaxations. A reference value of approximately **-1.90 eV/atom** is expected for these settings (consistent with Materials Project PBE calculations). A helper function `compute_dft_voltage(e_lithiated, e_delithiated, n_li, e_li_metal)` will be included in `dft_prep.py`. This function is in scope; the user provides the total energies parsed from VASP OUTCARs.

---

## 5. Notebook (`cathode_filtering.ipynb`)

Three sections:

1. **Load & Filter** — execute `screen_cathode` filtering logic inline, display ranked table with all metrics
2. **Visualize** — histograms of `avg_voltage`, `max_grav_cap`, `ehull`; scatter plot of voltage vs. capacity colored by composite score
3. **Select & Export** — user defines `selected_jids = [...]`, calls `dft_prep.generate_inputs(selected_jids, output_dir=None)` to write VASP input directories. `output_dir` defaults to `batterymat/screening_cathode/dft_inputs/` when `None`. Returns a list of created directory paths.

---

## 6. Dependencies

All dependencies are already present in the project environment:

| Package | Use |
|---------|-----|
| `jarvis-tools` | Structure handling, JARVIS-DFT data, vacancy generation |
| `alignn` | ALIGNN force field for vacancy energy ranking |
| `pandas` | Data filtering and ranking |
| `numpy` | Score normalization |
| `matplotlib` | Notebook visualizations |

No new dependencies are required.

---

## 7. Out of Scope

- Re-running ALIGNN screening from scratch (Stage 1 uses existing `Li_min.csv`)
- NEB migration barrier calculations (covered by existing `neb_calc/`)
- Multi-ion cathode screening (Na, K, etc.) — Li only in this iteration
- Automatic POTCAR generation (requires licensed VASP)
- Post-DFT analysis automation beyond the voltage helper function (e.g., automated OUTCAR parsing, bandgap extraction from DOSCAR)
