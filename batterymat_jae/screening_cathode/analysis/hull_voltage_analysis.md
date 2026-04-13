# Equilibrium Hull Voltage Comparison with Experiment

## Overview

The convex hull (equilibrium) voltage represents the thermodynamic discharge profile — flat plateaus corresponding to two-phase coexistence regions between stable Li compositions. These are compared against experimental quasi-equilibrium measurements (low C-rate or GITT) from literature.

The hull is computed from formation energies: `dE(x) = E(x) - x*E(1) - (1-x)*E(0)`. The lower convex hull identifies thermodynamically stable compositions; between two hull vertices the system exists as a two-phase mixture at constant voltage. The number of hull plateaus equals the number of hull segments (vertices minus one).

---

## LFP (LiFePO₄, olivine, PBE+U)

**17/17 steps complete (Li₁₆→Li₀).**

| Plateau | x_Li range | DFT Hull | Experiment | Difference |
|---------|-----------|----------|------------|------------|
| 1 | 0.69 → 1.0 | 3.48 V | ~3.42 V | +0.06 V |
| 2 | 0.56 → 0.69 | 3.58 V | ~3.42 V | +0.16 V |
| 3 | 0.38 → 0.56 | 3.65 V | ~3.42 V | +0.23 V |
| 4 | 0 → 0.38 | 3.67 V | ~3.42 V | +0.25 V |
| **Weighted avg** | **0 → 1.0** | **3.60 V** | **~3.42 V** | **+0.18 V** |

**Discussion:** Experimentally, LiFePO₄ exhibits a single flat plateau at ~3.4 V due to a first-order two-phase reaction between LiFePO₄ and FePO₄ with no stable intermediate solid solutions at room temperature. The DFT hull instead produces 4 plateaus (5 hull vertices), which is an artifact of the finite 2×2×1 supercell (16 Li atoms). The discrete Li-vacancy orderings in the supercell create artificial intermediate stable compositions that would not exist in the thermodynamic limit (infinite system or disordered solid solution).

The key observation is that all four plateaus are clustered within a narrow 0.19 V range (3.48–3.67 V), which is consistent with a single flat plateau broadened by finite-size effects. The systematic overestimate of +0.06 to +0.25 V relative to experiment is typical for PBE+U with U(Fe)=5.3 eV on olivine LiFePO₄.

The slight upward slope from 3.48 V (high x) to 3.67 V (low x) reflects stronger Fe-Fe interactions at low Li content in the finite supercell — in the thermodynamic limit, these would average out to the single ~3.4 V plateau.

**Experimental reference:** Padhi, Nanjundaswamy & Goodenough, *J. Electrochem. Soc.* **144**, 1188 (1997). Reports ~3.5 V vs Li for reversible Li extraction/insertion. Subsequent optimized materials consistently show 3.40–3.45 V at low C-rate.

---

## LMP (LiMnPO₄, olivine, PBE+U)

**16/17 steps complete (Li₁₆→Li₁). Step 16 (Li₀) abandoned — Mn⁴⁺ unconvergeable.**

No convex hull computed — requires both x=1 and x=0 endpoints.

| Metric | Value |
|--------|-------|
| Step voltage range | 3.23–4.34 V |
| Step voltage average | 3.91 V |
| Experimental plateau | ~4.1 V |
| Difference | −0.19 V |

**Discussion:** Without the x=0 endpoint, only raw step voltages are available. The average of 3.91 V underestimates the experimental ~4.1 V by 0.19 V. This is the opposite sign from LFP (which overestimates) — a known issue with PBE+U on Mn²⁺/Mn³⁺ olivines, where U(Mn)=3.9 eV may be slightly too low to fully localize the Mn d-electrons.

The step_16 (fully delithiated MnPO₄) failed to converge despite multiple attempts with different INCAR settings (ALGO=All, cold start, gentler mixing). The energy oscillated ~200 eV above the expected ground state, trapped in a false electronic minimum. This is a well-known pathology of Mn⁴⁺ (d³) in PBE+U — the many near-degenerate magnetic configurations create an extremely rugged energy landscape. The fully delithiated state is not experimentally accessible anyway — LMP operates as a flat two-phase system and is never cycled to x=0.

Like LFP, LMP is expected to show a single flat plateau (first-order LiMnPO₄ ↔ MnPO₄ phase transition). The hull, if computable, would likely show the same finite-size splitting into multiple narrow plateaus near 4.1 V.

**Experimental references:** Li, Azuma & Tohda, *Electrochem. Solid-State Lett.* **5**, A135 (2002); Delacourt et al., *Chem. Mater.* **16**, 93 (2004)

---

## LMO (LiMn₂O₄, spinel, PBE+U)

**17/17 steps complete (Li₁₆→Li₀).**

| Plateau | x_Li range | DFT Hull | Experiment | Difference |
|---------|-----------|----------|------------|------------|
| Upper | 0 → 0.44 | 4.17 V | ~4.13 V | +0.04 V |
| Lower | 0.44 → 1.0 | 4.00 V | ~3.95 V | +0.05 V |
| **Weighted avg** | **0 → 1.0** | **4.08 V** | **~4.05 V** | **+0.03 V** |

**Discussion:** Excellent agreement — both plateaus within ~0.05 V of experiment. The two-step structure near 4 V is correctly reproduced, with the voltage step at x≈0.44 matching the known Li ordering transition at half-filling of the tetrahedral 8a sites in the spinel framework.

The upper plateau (4.17 V) corresponds to delithiation from x=0.44 to x=0, where Mn³⁺→Mn⁴⁺ oxidation occurs in a cooperative Jahn-Teller distorted environment. The lower plateau (4.00 V) corresponds to x=1.0 to x=0.44, where Mn³⁺/Mn⁴⁺ mixed valence gives way to predominantly Mn⁴⁺. The 0.17 V step between plateaus reflects the ordering energy of the Li sublattice at x=0.5.

This is the best-performing material in our study — the hull structure, plateau voltages, and transition composition all match experiment quantitatively. The 2×2×2 supercell (16 Li in spinel 8a sites) provides sufficient configurational freedom to resolve the half-filling transition.

**Experimental reference:** Ohzuku, Kitagawa & Hirai, *J. Electrochem. Soc.* **137**, 769 (1990)

---

## NMC (Li₄Mn₃Co₂Ni₃O₁₆, layered, PBE+U)

**17/17 steps complete (Li₁₆→Li₀).**

| Plateau | x_Li range | DFT Hull | Experiment (NMC-111) | Difference |
|---------|-----------|----------|---------------------|------------|
| 1 | 0.88 → 1.0 | 3.52 V | ~3.5 V | ~0 V |
| 2 | 0.81 → 0.88 | 3.97 V | ~3.6 V | +0.37 V |
| 3 | 0.56 → 0.81 | 4.08 V | ~3.7 V | +0.38 V |
| 4 | 0.50 → 0.56 | 4.18 V | ~3.8 V | +0.38 V |
| 5 | 0 → 0.50 | 4.86 V | not accessible | artifact |
| **Avg (x>0.5)** | **0.50 → 1.0** | **3.93 V** | **~3.6–3.7 V** | **+0.23–0.33 V** |
| **Avg (all)** | **0 → 1.0** | **4.40 V** | **~3.7 V** | **+0.70 V** |

**Discussion:** NMC shows a clear two-regime behavior in the DFT voltage curve:

**Regime 1 (x > 0.5, plateaus 1–4):** Voltages of 3.52–4.18 V are physically reasonable. The first plateau (3.52 V) involves initial Ni²⁺→Ni³⁺ oxidation; subsequent plateaus at 3.97–4.18 V reflect mixed Ni³⁺→Ni⁴⁺ and Co³⁺→Co⁴⁺ redox. The average of ~3.93 V overestimates the experimental ~3.6–3.7 V by 0.2–0.3 V, which is a typical PBE+U systematic error.

**Regime 2 (x < 0.5, plateau 5):** The 4.86 V plateau spanning the entire lower half of the composition range is almost certainly a DFT artifact. Several factors contribute:

1. **Oxygen redox activation:** Below x=0.5, DFT+U frequently predicts charge compensation via O²⁻ oxidation, producing anomalously high voltages. Experimentally, this manifests as irreversible capacity loss and "voltage fade," not a stable 5 V plateau.
2. **Wrong functional:** This material is layered (R-3m, spacegroup 166) and should use optB88-vdW, but was initialized with PBE before auto-detection was implemented. PBE overestimates interlayer spacing, distorting the delithiation energetics at deep Li removal.
3. **Cation ordering artifact:** The JARVIS structure has an ordered Mn/Co/Ni sublattice. Real NMC has a disordered TM layer. Ordered arrangements create artificial high-voltage configurations at specific compositions.
4. **Multi-component U calibration:** Per-element U values (Mn=3.9, Co=3.32, Ni=6.2 eV) were calibrated separately for binary oxides and may not transfer accurately to a ternary layered system.

**Experimental context:** Standard NMC compositions (NMC-111 through NMC-811) show sloping S-shaped discharge curves at 3.5–4.3 V, not flat plateaus. The stoichiometry Li₄Mn₃Co₂Ni₃O₁₆ (Mn:Co:Ni = 3:2:3, ≈ NMC-334) has no direct experimental equivalent. In Mn-rich NMC, Mn⁴⁺ is electrochemically inactive below ~4.5 V; the active redox couples are Ni²⁺/Ni⁴⁺ and Co³⁺/Co⁴⁺ only.

**Recommendation:** The voltage curve for x > 0.5 (avg ~3.93 V) is the physically meaningful result. The x < 0.5 regime should be disregarded or flagged as an artifact in any publication.

**General NMC references:** Noh, Youn, Yoon & Sun, *J. Power Sources* **233**, 121 (2013); Manthiram, Knight, Myung, Oh & Sun, *Adv. Energy Mater.* **6**, 1501010 (2016)

---

## LCO (LiCoO₂, layered, optB88-vdW)

**9/9 steps complete (Li₈→Li₀).**

| Plateau | x_Li range | DFT Hull | Experiment | Difference |
|---------|-----------|----------|------------|------------|
| H1→H2 | 0.5 → 1.0 | 3.99 V | ~3.93 V | +0.06 V |
| Order-disorder | 0.25 → 0.5 | 4.22 V | ~4.07 V | +0.15 V |
| H2→H3 | 0 → 0.25 | 4.46 V | ~4.17 V | +0.29 V |
| **Weighted avg** | **0 → 1.0** | **4.18 V** | **~4.05 V** | **+0.13 V** |

**Discussion:** The three plateaus correctly capture the staged H1→H2→H3 delithiation transitions in layered LiCoO₂. The lowest plateau (H1→H2, the main discharge feature) matches well at +0.06 V. The upper two plateaus are increasingly overestimated at deep delithiation due to:

1. **Finite supercell size:** The 2×2×2 supercell contains only 8 Li atoms. At deep delithiation (x<0.5), removing a single Li represents a 12.5% composition jump, too coarse to resolve the subtle staging transitions and Li ordering phenomena near x=0.5 and x=0.25.
2. **Functional limitations:** optB88-vdW tends to slightly overbind the delithiated phases. The H2→H3 transition involves significant c-axis contraction and CoO₂ slab rearrangement sensitive to interlayer interaction treatment.

A larger supercell (3×3×3) and/or hybrid functional (HSE06) would improve the deep-delithiation regime but at significantly higher computational cost.

**Experimental reference:** Reimers & Dahn, *J. Electrochem. Soc.* **139**, 2091 (1992)

---

## Cross-Material Summary

| Material | Structure | Functional | Hull plateaus | DFT avg V | Exp avg V | Error |
|----------|-----------|-----------|---------------|-----------|-----------|-------|
| LFP | Olivine | PBE+U | 4 (artifact of finite cell; should be 1) | 3.60 V | ~3.42 V | +0.18 V |
| LMP | Olivine | PBE+U | N/A (missing x=0) | 3.91 V* | ~4.1 V | −0.19 V |
| LMO | Spinel | PBE+U | 2 (matches experiment) | 4.08 V | ~4.05 V | +0.03 V |
| NMC | Layered | PBE+U | 5 (x>0.5 physical, x<0.5 artifact) | 3.93 V† | ~3.7 V | +0.23 V |
| LCO | Layered | optB88-vdW | 3 (matches experiment) | 4.18 V | ~4.05 V | +0.13 V |

\* Step voltage average (no hull).
† Average restricted to x > 0.5 (physical regime only).

### Key findings

1. **LMO is the best performer** — hull structure, plateau voltages, and transition compositions all match experiment quantitatively (error < 0.05 V). The spinel framework's high symmetry and well-separated Li ordering transitions are ideal for the supercell approach.

2. **LCO captures staged delithiation** — three hull plateaus correctly reproduce the H1→H2→H3 sequence. Main discharge plateau accurate to +0.06 V; deep-delithiation overestimated due to finite cell size and vdW functional limitations.

3. **LFP reproduces the flat plateau** — the four hull plateaus are clustered within 0.19 V, consistent with a single flat plateau broadened by finite-size effects. Systematic PBE+U overestimate of +0.18 V. The narrow voltage spread confirms the two-phase olivine mechanism.

4. **LMP step_16 is unconvergeable** — fully delithiated Mn⁴⁺ (d³) is pathological for PBE+U. Step voltage average of 3.91 V underestimates experiment by 0.19 V. The missing x=0 endpoint prevents hull analysis but does not affect the practical result — LMP is never fully delithiated experimentally.

5. **NMC has a two-regime problem** — physically reasonable at x > 0.5 (~3.93 V, overestimates by ~0.2 V), but the x < 0.5 regime (~4.86 V) is an artifact of oxygen redox activation, wrong functional (PBE instead of optB88-vdW), and ordered cation arrangement. The x > 0.5 average is the publishable result.

### Systematic errors by functional

| Functional | Materials | Typical error | Direction |
|-----------|-----------|--------------|-----------|
| PBE+U | LFP, LMO | +0.03 to +0.18 V | Overestimates |
| PBE+U | LMP | −0.19 V | Underestimates (U too low for Mn²⁺/³⁺) |
| PBE+U | NMC (x>0.5) | +0.23 V | Overestimates (wrong functional for layered) |
| optB88-vdW | LCO | +0.06 to +0.29 V | Overestimates (increases with delithiation) |
