#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data, get_jid_data
from jarvis.core.specie import Specie
from jarvis.io.vasp.inputs import Poscar
from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
from jarvis.core.atoms import get_supercell_dims
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(message, log_file=None):
    """Print and optionally write to log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')


def load_database_cached(cache_file='jarvis_dft3d_cache.pkl', log_file=None):
    """Load JARVIS database with caching to avoid repeated downloads"""
    if os.path.exists(cache_file):
        log(f"Loading cached database from {cache_file}", log_file)
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        log("Downloading JARVIS-DFT database (first time, ~5 minutes)...", log_file)
        dft3d = data('dft_3d')
        log(f"Caching database to {cache_file}", log_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(dft3d, f)
        return dft3d


# ============================================================================
# CAPACITY CALCULATIONS
# ============================================================================

def calculate_gravimetric_capacity(n_electrons, molar_mass):
    """Calculate gravimetric capacity in mAh/g"""
    F = 96485.3329 / 3600  # Faraday's constant in mAh/mol
    return 1000 * (n_electrons * F) / molar_mass


def calculate_volumetric_capacity(grav_capacity, density):
    """Calculate volumetric capacity in mAh/cm³"""
    return grav_capacity * density


def get_theoretical_capacity(atoms, n_electrons=1):
    """Calculate theoretical capacities (returns tuple: vol, grav)"""
    amu_to_g = 1.66054e-24
    molar_mass = atoms.composition.weight
    volume = atoms.volume  # Å³
    density = molar_mass * amu_to_g / (volume * 1e-24)  # g/cm³

    grav_cap = calculate_gravimetric_capacity(n_electrons, molar_mass)
    vol_cap = calculate_volumetric_capacity(grav_cap, density)

    return vol_cap, grav_cap


# ============================================================================
# ENERGY CALCULATIONS
# ============================================================================

def calculate_energy(calculator, atoms, relax_cell=False,
                    constant_volume=False, fmax=0.05, nsteps=100):
    """Calculate energy using ALIGNN calculator"""
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    if not relax_cell:
        return ase_atoms.get_potential_energy()

    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)

    return ase_atoms.atoms.get_potential_energy()


# ============================================================================
# VOLTAGE PROFILE CALCULATION
# ============================================================================

def get_voltage_profile(calculator, atoms, element='Li', charge=None, length=8):
    """
    Calculate voltage profile by systematically removing ions
    Memory-optimized: doesn't store full structure dictionaries
    """
    if charge is None:
        charge = abs(Specie(element).element_property('min_oxid_s'))

    # Create supercell
    dim = get_supercell_dims(atoms, enforce_c_size=length, extend=1)
    atoms = atoms.make_supercell(dim)

    # Get reference chemical potential
    chem_pot_data = get_optb88vdw_energy()
    jid_elemental = chem_pot_data[element]["jid"]
    atoms_elemental = Atoms.from_dict(
        get_jid_data(jid=jid_elemental, dataset="dft_3d")["atoms"]
    )

    atoms_en = (calculate_energy(calculator, atoms_elemental, relax_cell=False)
                / atoms_elemental.num_atoms)
    chem_pot = charge * atoms_en

    # Identify removable ions
    elements = atoms.elements
    coords = atoms.frac_coords
    lattice_mat = atoms.lattice_mat

    ion_indices = [i for i, el in enumerate(elements) if el == element]

    # Generate removal combinations
    combinations = []
    for i in range(len(ion_indices)):
        combinations.append(ion_indices[0:i+1])

    # Calculate energies
    results = []
    full_energy = calculate_energy(atoms=atoms, calculator=calculator)
    results.append({
        'en': full_energy,
        'voltage': None,
        'comp': 1.0,
        'n_removed': 0
    })

    for comb in combinations:
        new_elements = []
        new_coords = []

        for j, el in enumerate(elements):
            if j not in comb:
                new_elements.append(el)
                new_coords.append(coords[j])

        new_atoms = Atoms(
            coords=new_coords,
            lattice_mat=lattice_mat,
            elements=new_elements,
            cartesian=False
        )

        energy = calculate_energy(atoms=new_atoms, calculator=calculator)
        voltage = (energy - results[-1]['en'] + atoms_en) / charge
        comp = 1.0 - (len(comb) / len(ion_indices))

        results.append({
            'en': energy,
            'voltage': voltage,
            'comp': comp,
            'n_removed': len(comb)
        })

    return results[1:]  # Remove first entry (no voltage)


# ============================================================================
# SINGLE MATERIAL ANALYSIS
# ============================================================================

def read_poscar_manual(poscar_path):
    """Manually read POSCAR file to avoid JARVIS parser bugs"""
    with open(poscar_path, 'r') as f:
        all_lines = f.readlines()
    
    # Remove comment lines and empty lines
    lines = [line for line in all_lines if line.strip() and not line.strip().startswith('#')]
    
    # Line 0: Comment
    # Line 1: Scaling factor
    scale = float(lines[1].strip())
    
    # Lines 2-4: Lattice vectors
    lattice = []
    for i in range(2, 5):
        vec = [float(x) * scale for x in lines[i].split()]
        lattice.append(vec)
    
    # Line 5: Element symbols
    elements_list = lines[5].split()
    
    # Line 6: Element counts
    counts = [int(x) for x in lines[6].split()]
    
    # Line 7: Direct or Cartesian
    coord_type = lines[7].strip()
    cartesian = coord_type.startswith('C') or coord_type.startswith('c')
    
    # Lines 8+: Coordinates
    coords = []
    elements = []
    line_idx = 8
    
    for element, count in zip(elements_list, counts):
        for _ in range(count):
            if line_idx >= len(lines):
                raise ValueError(f"Not enough coordinate lines in POSCAR. Expected {sum(counts)} atoms, found {len(coords)}")
            
            coord_line = lines[line_idx].split()
            if len(coord_line) < 3:
                raise ValueError(f"Invalid coordinate line at index {line_idx}: {lines[line_idx]}")
            
            coords.append([float(coord_line[0]), float(coord_line[1]), float(coord_line[2])])
            elements.append(element)
            line_idx += 1
    
    # Create JARVIS Atoms object
    atoms = Atoms(
        lattice_mat=lattice,
        coords=coords,
        elements=elements,
        cartesian=cartesian
    )
    
    return atoms


def analyze_single_material(jid=None, poscar_path=None, calculator=None, 
                           element='Li', output_dir='results'):
    """Analyze single material from JID or POSCAR file and save plot"""
    os.makedirs(output_dir, exist_ok=True)

    # Load structure from JID or POSCAR
    if poscar_path:
        log(f"Loading structure from POSCAR: {poscar_path}")
        atoms = read_poscar_manual(poscar_path)
        
        material_id = os.path.basename(poscar_path).replace('.vasp', '').replace('POSCAR', 'material')
        if material_id == 'material':
            material_id = os.path.basename(os.path.dirname(poscar_path))
    elif jid:
        log(f"Loading structure from JARVIS: {jid}")
        data_dict = get_jid_data(jid=jid, dataset='dft_3d')
        atoms = Atoms.from_dict(data_dict["atoms"])
        material_id = jid
    else:
        raise ValueError("Either jid or poscar_path must be provided")

    # Calculate voltage profile
    voltage_profile = get_voltage_profile(
        calculator=calculator,
        atoms=atoms,
        element=element
    )

    # Extract data
    voltages = [entry['voltage'] for entry in voltage_profile]
    compositions = np.array([entry['comp'] for entry in voltage_profile])

    # Calculate capacities
    vol_cap, grav_cap = get_theoretical_capacity(atoms)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(compositions, voltages, '-o', linewidth=2, markersize=8)
    plt.xlabel('Ion Fraction', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title(f'{atoms.composition.reduced_formula} ({material_id})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{material_id}_voltage.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compile results
    results = {
        'material_id': material_id,
        'formula': atoms.composition.reduced_formula,
        'voltage_profile': voltage_profile,
        'voltages': voltages,
        'compositions': compositions.tolist(),
        'max_voltage': max(voltages),
        'min_voltage': min(voltages),
        'avg_voltage': np.mean(voltages),
        'volumetric_capacity': vol_cap,
        'gravimetric_capacity': grav_cap,
        'status': 'success'
    }

    return results


# ============================================================================
# HIGH-THROUGHPUT SCREENING
# ============================================================================

def screen_materials(start_idx=0, end_idx=None, output_file='results.pkl',
                    checkpoint_file='checkpoint.pkl', log_file='screening.log',
                    cache_file='jarvis_dft3d_cache.pkl'):
    """
    start_idx: Starting index in filtered database
    end_idx: Ending index (None = process all)
    output_file: Output pickle file
    checkpoint_file: Checkpoint file to track progress
    log_file: Log file path
    cache_file: Database cache file
    """
    log(f"Starting screening from index {start_idx} to {end_idx}", log_file)

    # Load processed JIDs from checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            processed_jids = checkpoint_data.get('processed_jids', set())
            results = checkpoint_data.get('results', [])
        log(f"Loaded checkpoint: {len(processed_jids)} materials already processed", log_file)
    else:
        processed_jids = set()
        results = []

    # Load database
    dft3d = load_database_cached(cache_file, log_file)
    df = pd.DataFrame(dft3d)

    # Filter for Li-containing intercalation materials
    log("Filtering database for Li intercalation materials...", log_file)
    df = df[df['dimensionality'] == 'intercalated ion']
    df['has_Li'] = df['atoms'].apply(lambda x: 'Li' in x['elements'])
    df = df[df['has_Li'] == True]

    # Apply index range
    if end_idx is None:
        end_idx = len(df)
    df = df.iloc[start_idx:end_idx]

    log(f"Processing {len(df)} materials", log_file)

    # Initialize calculator
    log("Initializing ALIGNN calculator...", log_file)
    calculator = AlignnAtomwiseCalculator(
        path=default_path(),
        stress_wt=0.3
    )

    # Process materials
    for idx, (i, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                        desc="Screening")):
        jid = row['jid']

        # Skip if already processed
        if jid in processed_jids:
            continue

        try:
            atoms = Atoms.from_dict(row['atoms'])

            voltage_profile = get_voltage_profile(
                calculator=calculator,
                atoms=atoms,
                element='Li'
            )

            voltages = [j['voltage'] for j in voltage_profile]
            vol_cap, grav_cap = get_theoretical_capacity(atoms)

            result = {
                'jid': jid,
                'formula': row['formula'],
                'ehull': row.get('ehull', None),
                'bandgap': row.get('optb88vdw_bandgap', None),
                'max_voltage': max(voltages),
                'min_voltage': min(voltages),
                'avg_voltage': np.mean(voltages),
                'voltages': voltages,
                'volumetric_capacity': vol_cap,
                'gravimetric_capacity': grav_cap,
                'status': 'success'
            }

            results.append(result)
            processed_jids.add(jid)

            log(f"[{idx+1}/{len(df)}] {jid} ({row['formula']}): "
                f"V_avg={result['avg_voltage']:.2f}V, "
                f"Cap={result['gravimetric_capacity']:.1f} mAh/g", log_file)

        except Exception as e:
            log(f"[{idx+1}/{len(df)}] FAILED {jid}: {str(e)}", log_file)
            results.append({
                'jid': jid,
                'formula': row['formula'],
                'status': 'failed',
                'error': str(e)
            })
            processed_jids.add(jid)

        # Save checkpoint every 10 materials
        if len(processed_jids) % 10 == 0:
            checkpoint_data = {
                'processed_jids': processed_jids,
                'results': results
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

    # Final save
    log(f"Saving final results to {output_file}", log_file)
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    log(f"Screening complete: {successful} successful, {failed} failed", log_file)

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Battery cathode screening for HPC clusters'
    )

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['single', 'screen'],
                       default='screen', help='Analysis mode')

    # Single material mode
    parser.add_argument('--jid', type=str, help='JARVIS ID for single material')
    parser.add_argument('--poscar', type=str, help='Path to POSCAR file')

    # Screening mode
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index in database')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='Ending index (None=all)')

    # Output files
    parser.add_argument('--output', type=str, default='results.pkl',
                       help='Output pickle file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pkl',
                       help='Checkpoint file')
    parser.add_argument('--log', type=str, default='screening.log',
                       help='Log file')
    parser.add_argument('--cache', type=str, default='jarvis_cache.pkl',
                       help='Database cache file')

    args = parser.parse_args()

    # Print job info
    log(f"Job started", args.log)
    log(f"Mode: {args.mode}", args.log)
    log(f"Hostname: {os.uname().nodename}", args.log)
    log(f"Working directory: {os.getcwd()}", args.log)

    if args.mode == 'single':
        # Check that either --jid or --poscar is provided
        if not args.jid and not args.poscar:
            log("ERROR: Either --jid or --poscar required for single mode", args.log)
            sys.exit(1)

        # Initialize calculator
        log("Initializing ALIGNN calculator...", args.log)
        calculator = AlignnAtomwiseCalculator(
            path=default_path(),
            stress_wt=0.3
        )
        
        # Run analysis
        result = analyze_single_material(
            jid=args.jid,
            poscar_path=args.poscar,
            calculator=calculator
        )

        log(f"Results for {result['formula']}:", args.log)
        log(f"  Avg Voltage: {result['avg_voltage']:.2f} V", args.log)
        log(f"  Capacity: {result['gravimetric_capacity']:.2f} mAh/g", args.log)

        with open(args.output, 'wb') as f:
            pickle.dump(result, f)

    else:  # screening mode
        results = screen_materials(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            output_file=args.output,
            checkpoint_file=args.checkpoint,
            log_file=args.log,
            cache_file=args.cache
        )

    log("Job completed successfully", args.log)


if __name__ == "__main__":
    main()
