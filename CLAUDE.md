# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BatteryMat** is a computational pipeline for high-throughput screening and analysis of battery electrode materials, developed at NIST. It uses ML force fields (ALIGNN) and DFT calculations to predict voltage profiles and ion migration barriers for candidate materials.

## Setup

```bash
pip install -e .
```

Key dependencies: `jarvis-tools`, `alignn`, `numpy`, `scipy`. Python >= 3.8 required.

## Architecture

The project is organized into five research stages:

1. **`screening_anode/`** — Core screening pipeline. `screen.py` loads structures from the JARVIS-DFT database (~76k materials), filters intercalated-ion structures, then calculates voltage by successively removing ion atoms and computing energies with the ALIGNN ML force field. Results export as JSON, then converted to CSV via `json_to_csv.py`. Results per ion type are stored as `[Ion].csv`.

2. **`average_voltage/`** — Post-screening filtering. `Battery_Filtering.ipynb` applies thresholds (max V ≤ 5.5 V, ehull ≤ 0.05 eV, min V > 1 V) to produce `[Ion]_min.csv` files.

3. **`screening_cathode/`** — Li-ion cathode screening and DFT preparation (added 2026-03-11). Two stages:
   - **Stage 1 (`screen_cathode.py`)**: Filters and ranks cathode candidates from `average_voltage/Li_min.csv` joined with JARVIS-DFT metadata. Filters: avg_voltage 3.0–4.5 V, max_grav_cap > 20 mAh/g, ehull ≤ 0.05 eV, max_voltage ≤ 5.5 V. Composite score: `(1/3)*norm(avg_voltage) + (1/3)*norm(max_grav_cap) - (1/3)*norm(ehull)`. Exports `cathode_candidates_ranked.csv`. Run as script: `python screen_cathode.py`.
   - **Stage 2 (`dft_prep.py`)**: For user-selected JIDs, generates 4 VASP input directories per candidate (PBE relax lithiated/delithiated, TB-mBJ static lithiated/delithiated). Delithiated structure selected via ALIGNN-ranked Wyckoff-inequivalent Li vacancies (`Vacancy.generate_defects(enforce_c_size=0.0, extend=1)`). Run as script: `python dft_prep.py JVASP-XXXXX [JVASP-YYYYY ...]`.
   - **`cathode_filtering.ipynb`**: Interactive notebook for filtering, visualization, and DFT export.

4. **`neb_calc/`** — Nudged Elastic Band (NEB) calculations for ion migration barriers. Contains POSCAR structure files for migration path images for two materials (NiMnO, V2O5) across multiple ion types.

5. **`benchmarks/`** — Benchmarking against known cathode materials (NMC, V2O5, TiSe2, MnO2, Chevrel phase). DFT results stored as POSCARs under `PBE/` and `PBE-D3/` subdirectories.

## Core Voltage Calculation

The central algorithm in `screen.py`:
```
V = (E_removed - E_full - unary_energy(ion)) / charge
```
Where energies are computed with the ALIGNN force field. Six ion types are supported: Li, Na, K, Ca, Mg, Al, Zn.

For post-DFT cathode voltage (`dft_prep.py`):
```
V = (E_delithiated - E_lithiated - n_li * E_li_metal) / n_li
```
`E_li_metal` ≈ -1.90 eV/atom (PBE, Li_sv PAW, ENCUT=520).

## VASP DFT Input Settings (screening_cathode)

**PBE relaxation:** IBRION=2, NSW=100, ISIF=3, ENCUT=520, EDIFF=1E-6, EDIFFG=-0.01, LCHARG=.TRUE.

**TB-mBJ static:** IBRION=-1, NSW=0, METAGGA=MBJ, LASPH=.TRUE., ICHARG=11, LORBIT=11, NEDOS=2000. Requires CHGCAR from preceding PBE run.

**KPOINTS:** Gamma-centered 3×3×3.

**POTCAR:** Materials Project PAW labels (Li_sv, Mn_pv, etc.). See `_MP_PAW` dict in `dft_prep.py`.

## ALIGNN Model Weights

ALIGNN uses `wt01_path()` which downloads `alignnff_wt01` weights from figshare. If the automated download fails (WAF block), manually place `best_model.pt` and `config.json` in:
```
/Users/jaelee/software/alignn/alignn/ff/alignnff_wt01/
```
The download is skipped if `best_model.pt` already exists there.

## max_grav_cap Units

The `max_grav_cap` column in `Li_min.csv` is normalized to the full unit cell mass (not per formula unit). LiCoO2 has ~36.5 mAh/g in this dataset vs. its theoretical ~274 mAh/g. The screening threshold is 20 mAh/g (not 100).

## Data

- Screened materials: ~7,474 structures for lithium intercalation
- External data (not in repo): https://drive.google.com/drive/u/2/folders/1vf42QPRKc_kFXo5kmnWvDKzRFErUDdU3
- Google Colab demo: https://colab.research.google.com/gist/knc6/e7e3aecf3b748c0df1d58dc7911da950/
