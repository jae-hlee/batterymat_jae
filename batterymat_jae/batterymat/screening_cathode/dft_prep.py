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


def get_delithiated_structure(atoms):
    """Return the lowest-energy single-Li-vacancy structure.

    Uses ALIGNN-ranked Wyckoff-inequivalent vacancy configurations.
    If n_Li == 1, returns the fully delithiated host directly.
    If n_Li == 0, raises ValueError.

    Args:
        atoms: jarvis.core.atoms.Atoms object (fully lithiated).

    Returns:
        jarvis.core.atoms.Atoms -- delithiated structure.
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
    defects = vac.generate_defects(enforce_c_size=0.0, extend=1)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate VASP DFT inputs for cathode candidates.")
    parser.add_argument("jids", nargs="+", help="JARVIS JIDs to process (e.g. JVASP-1234)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: dft_inputs/)")
    args = parser.parse_args()
    generate_inputs(args.jids, output_dir=args.output_dir)
