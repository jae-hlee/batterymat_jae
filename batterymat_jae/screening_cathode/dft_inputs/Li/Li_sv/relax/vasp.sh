#!/bin/bash
export OMP_NUM_THREADS=1
export VASP_VDW_KERNEL=/home/jlee859/vdw_kernel/vdw_kernel.bindat
mpirun -np 16 /opt/vasp/bin/vasp_std
