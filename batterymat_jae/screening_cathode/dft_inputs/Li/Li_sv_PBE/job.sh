#!/bin/bash
#SBATCH --job-name=Li_sv
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=100G
#SBATCH --partition=main
#SBATCH --output=vasp_%j.out
#SBATCH --error=vasp_%j.err

export OMP_NUM_THREADS=1
export VASP_VDW_KERNEL=/home/jlee859/vdw_kernel/vdw_kernel.bindat
mpirun -np 64 /opt/vasp/bin/vasp_std
