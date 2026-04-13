#!/bin/bash

export OMP_NUM_THREADS=1
mpirun -np 16 /opt/vasp/bin/vasp_std
