# Electromagnetic Applications

This directory contains electromagnetic field simulation examples using H(curl) Nedelec finite elements.

## Maxwell PML Point Source

`maxwell_pml_point_source.py` - 2D Maxwell equations with Perfectly Matched Layer (PML) absorbing boundary conditions and dipole point source excitation.

**Features:**
- H(curl) Nedelec edge elements for electric field
- Complex-valued frequency-domain formulation
- PML absorbing boundaries to simulate open domains
- HYPRE AMS preconditioner for efficient solving
- Multi-threaded parallel acceleration with OpenMP

**Solver options:**
- `hypre_ams`: HYPRE AMS + GMRES (parallel, recommended for large problems)
- `pardiso`: Intel MKL PARDISO (multi-threaded direct solver)
- `direct`: scipy spsolve (single-threaded)

**Usage:**
```bash
# Single-threaded
python applications/electromagnetics/maxwell_pml_point_source.py

# Multi-threaded (28 cores)
export OMP_NUM_THREADS=28
python applications/electromagnetics/maxwell_pml_point_source.py
```

**Requirements:**
- PETSc 3.21+ with HYPRE 2.31+ (for hypre_ams solver)
- OpenMP-enabled HYPRE for multi-threading
- mpi4py, petsc4py
