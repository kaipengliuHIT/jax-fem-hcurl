# Pull Request: Add H(curl) Nedelec Edge Element Support

## Summary

This PR extends JAX-FEM with support for **H(curl) conforming Nedelec edge elements**, enabling the solution of **Maxwell's equations** and other electromagnetic problems.

## Features Added

### 1. Core H(curl) Implementation (`jax_fem/hcurl_fe.py`)
- `HCurlFiniteElement` class for edge-based DOF management
- Support for Nedelec first kind (N1E) and second kind (N2E) elements
- 2D elements: `N1E_TRI1`, `N1E_TRI2`, `N2E_TRI1`
- 3D elements: `N1E_TET1`, `N1E_TET2`, `N2E_TET1`
- Proper covariant Piola transformation for physical mapping
- Curl computation in physical coordinates
- Matrix assembly functions:
  - `compute_curl_curl_matrix()` - for curl-curl operator
  - `compute_mass_matrix()` - for mass terms
  - `compute_discrete_gradient()` - for AMS preconditioner

### 2. Basis Function Support (`jax_fem/basis.py`)
- Extended `get_elements()` to support Nedelec element families
- Added `is_hcurl_element()` utility function
- Proper handling of vector-valued shape functions

### 3. HYPRE AMS Preconditioner (`jax_fem/solver.py`)
- Integration of HYPRE's Auxiliary-space Maxwell Solver (AMS)
- Support for discrete gradient matrix input
- Node coordinate passing for optimal AMS performance
- **Important**: AMS requires GMRES (not CG) for reliable convergence

### 4. Examples
| Example | Description |
|---------|-------------|
| `maxwell_ams_example.py` | 2D Maxwell with N1E_TRI1 elements |
| `maxwell_3d_ams_example.py` | 3D Maxwell with N1E_TET1 tetrahedra |
| `maxwell_waveguide_example.py` | Mixed boundary conditions (PEC, incident wave) |
| `maxwell_pml_point_source.py` | Point source with PML absorbing boundary |
| `maxwell_solver_benchmark.py` | Solver performance comparison |

### 5. Unit Tests (`tests/test_hcurl_elements.py`)
- 14 comprehensive tests covering:
  - Element creation and DOF counting
  - Shape function evaluation
  - Curl computation
  - Matrix assembly
  - Boundary condition handling
  - Discrete gradient construction

## Performance Benchmark Results

Comparison of preconditioners for 2D Maxwell (N1E_TRI1):

| Mesh | DOFs | AMS (GMRES) | BoomerAMG | ILU | Jacobi |
|------|------|-------------|-----------|-----|--------|
| 4×4 | 32 | 8 iter | 8 iter | 8 iter | 20 iter |
| 8×8 | 208 | 19 iter | 17 iter | 14 iter | 51 iter |
| 12×12 | 456 | 30 iter | 25 iter | 21 iter | 76 iter |
| 16×16 | 800 | 43 iter | 32 iter | 27 iter | 102 iter |

**Key finding**: HYPRE AMS requires GMRES for reliable convergence (CG fails on larger meshes).

## Usage Example

```python
from jax_fem.hcurl_fe import (
    HCurlFiniteElement,
    compute_curl_curl_matrix,
    compute_mass_matrix,
    compute_discrete_gradient
)
from jax_fem.solver import petsc_solve

# Create H(curl) finite element space
fe = HCurlFiniteElement(
    mesh=mesh,
    dim=2,
    ele_type='N1E_TRI1',
    gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info
)

# Assemble matrices
K = compute_curl_curl_matrix(fe)  # curl-curl term
M = compute_mass_matrix(fe)       # mass term
A = K + omega**2 * M              # Maxwell system

# Solve with HYPRE AMS
G = compute_discrete_gradient(fe)
solver_options = {
    'petsc_solver': {
        'ksp_type': 'gmres',  # Must use GMRES, not CG!
        'pc_type': 'hypre',
        'hypre_type': 'ams',
        'discrete_gradient': G,
        'coordinates': mesh.points,
    }
}
x = petsc_solve(A_petsc, b, solver_options)
```

## Commits

1. `af2403e` - Core H(curl) Nedelec edge element support
2. `c3ff757` - HYPRE AMS preconditioner integration + Maxwell example
3. `8bfe31f` - README documentation update
4. `e42ca7d` - 3D Maxwell example with N1E_TET1
5. `8b4e9b0` - Waveguide example with mixed BCs
6. `1acd148` - Solver performance benchmark suite
7. `dafc74a` - Fix AMS convergence (use GMRES)
8. `5173fc4` - PML point source example

## Testing

```bash
# Run unit tests
pytest tests/test_hcurl_elements.py -v

# Run examples
python examples/maxwell_ams_example.py
python examples/maxwell_3d_ams_example.py
python examples/maxwell_waveguide_example.py
python examples/maxwell_pml_point_source.py
python examples/maxwell_solver_benchmark.py
```

## Dependencies

No new dependencies required. Uses existing:
- `basix` for Nedelec element definitions
- `petsc4py` with HYPRE for AMS solver
- `scipy` for sparse matrix operations

## Breaking Changes

None. All changes are additive.

## Related Issues

This PR addresses the need for electromagnetic simulation capabilities in JAX-FEM, enabling:
- Time-harmonic Maxwell equations
- Eddy current problems
- Waveguide analysis
- Scattering problems with PML

## Checklist

- [x] Code follows project style guidelines
- [x] Unit tests added and passing
- [x] Documentation updated (README)
- [x] Examples provided and working
- [x] No breaking changes to existing API
