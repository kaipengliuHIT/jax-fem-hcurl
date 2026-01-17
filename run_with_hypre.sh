#!/bin/bash
# Run JAX-FEM with HYPRE AMS support
# This script sets up the environment for using PETSc + HYPRE

export PATH=/home/kaipeng/code/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/home/kaipeng/code/opt/openmpi/lib:/home/kaipeng/code/opt/petsc/lib:/home/kaipeng/code/opt/hypre/lib:/home/kaipeng/code/opt/openblas/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/d/pythoncode/jax-fem/jax-fem:$PYTHONPATH

# Set OpenMP threads for shared-memory parallelism
# HYPRE uses OpenMP for intra-node parallelism
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}  # Use all 28 cores
export OPENBLAS_NUM_THREADS=28
export MKL_NUM_THREADS=28

# Run with MPI (for parallel execution)
# Usage: ./run_with_hypre.sh [script.py] [num_procs]
# Example: ./run_with_hypre.sh examples/maxwell_pml_point_source.py 4

SCRIPT=${1:-"examples/maxwell_pml_point_source.py"}
NP=${2:-1}

echo "OpenMP threads: $OMP_NUM_THREADS"
echo "MPI processes: $NP"

if [ $NP -gt 1 ]; then
    echo "Running with $NP MPI processes x $OMP_NUM_THREADS OpenMP threads..."
    mpirun -np $NP /home/kaipeng/miniconda3/envs/jax-fem/bin/python "$SCRIPT"
else
    echo "Running single MPI process with $OMP_NUM_THREADS OpenMP threads..."
    /home/kaipeng/miniconda3/envs/jax-fem/bin/python "$SCRIPT"
fi
