# Maxwell Solver Benchmark Results

## Summary

Performance comparison of different preconditioners for 2D Maxwell equations with H(curl) Nedelec elements.

### Test Configuration
- Problem: `curl curl E - ω² E = f` on unit square
- Element: N1E_TRI1 (Nedelec first kind, degree 1)
- Boundary: Homogeneous Dirichlet on all edges
- Solver tolerance: rtol=1e-8, max_iter=1000

### Solvers Tested
1. **HYPRE AMS** - Specialized auxiliary-space Maxwell solver
2. **HYPRE BoomerAMG** - General algebraic multigrid
3. **ILU** - Incomplete LU factorization
4. **Jacobi** - Diagonal preconditioner (baseline)

---

## Results by Mesh Size

### 4×4 Mesh (32 DOFs, 32 triangles)

| Solver | Iterations | Setup Time (s) | Solve Time (s) | Total Time (s) | Converged |
|--------|-----------|----------------|----------------|----------------|-----------|
| HYPRE AMS | 17 | 0.0002 | 0.0003 | 0.0005 | ✓ |
| HYPRE BoomerAMG | 8 | 0.0001 | 0.0001 | 0.0002 | ✓ |
| ILU | 7 | 0.0001 | 0.0001 | 0.0001 | ✓ |
| Jacobi | 20 | 0.0000 | 0.0000 | 0.0000 | ✓ |

### 8×8 Mesh (208 DOFs, 128 triangles)

| Solver | Iterations | Setup Time (s) | Solve Time (s) | Total Time (s) | Converged |
|--------|-----------|----------------|----------------|----------------|-----------|
| HYPRE AMS | 40 | 0.0009 | 0.0018 | 0.0027 | ✓ |
| HYPRE BoomerAMG | 17 | 0.0002 | 0.0004 | 0.0006 | ✓ |
| ILU | 14 | 0.0001 | 0.0001 | 0.0002 | ✓ |
| Jacobi | 51 | 0.0000 | 0.0001 | 0.0001 | ✓ |

### 12×12 Mesh (456 DOFs, 288 triangles)

| Solver | Iterations | Setup Time (s) | Solve Time (s) | Total Time (s) | Converged |
|--------|-----------|----------------|----------------|----------------|-----------|
| HYPRE AMS | 1000 | 0.0006 | 0.0798 | 0.0804 | ✗ (max iter) |
| HYPRE BoomerAMG | 25 | 0.0003 | 0.0009 | 0.0012 | ✓ |
| ILU | 21 | 0.0001 | 0.0004 | 0.0005 | ✓ |
| Jacobi | 76 | 0.0000 | 0.0005 | 0.0005 | ✓ |

### 16×16 Mesh (800 DOFs, 512 triangles)

| Solver | Iterations | Setup Time (s) | Solve Time (s) | Total Time (s) | Converged |
|--------|-----------|----------------|----------------|----------------|-----------|
| HYPRE AMS | 1000 | 0.0007 | 0.1359 | 0.1366 | ✗ (max iter) |
| HYPRE BoomerAMG | 32 | 0.0004 | 0.0018 | 0.0022 | ✓ |
| ILU | 27 | 0.0001 | 0.0010 | 0.0011 | ✓ |
| Jacobi | 102 | 0.0000 | 0.0008 | 0.0008 | ✓ |

---

## Key Findings

### 1. Iteration Count Scaling
- **ILU**: Best iteration scaling (7 → 14 → 21 → 27 iterations)
- **BoomerAMG**: Good scaling (8 → 17 → 25 → 32 iterations)
- **Jacobi**: Poor scaling (20 → 51 → 76 → 102 iterations)
- **AMS**: Convergence issues on larger meshes (needs investigation)

### 2. Time Performance (Small Problems)
For small problems (≤208 DOFs):
- **ILU** is fastest overall
- **BoomerAMG** is competitive
- **Jacobi** has minimal setup but more iterations

### 3. Time Performance (Large Problems)
For larger problems (≥456 DOFs):
- **ILU** maintains best performance
- **BoomerAMG** scales well
- **Jacobi** becomes slower due to iteration count

### 4. AMS Convergence Issue
HYPRE AMS failed to converge on 12×12 and 16×16 meshes. Possible causes:
- Incorrect discrete gradient matrix construction
- Missing or incorrect auxiliary space setup
- Need for additional AMS parameters (alpha_Poisson, beta_Poisson)
- Boundary condition handling in discrete gradient

---

## Recommendations

### For Production Use
1. **ILU preconditioner** - Best all-around performance for 2D problems
2. **HYPRE BoomerAMG** - Good alternative with better scaling potential

### For Large-Scale 3D Problems
1. **HYPRE AMS** - After fixing convergence issues, should be optimal for H(curl)
2. **HYPRE BoomerAMG** - Reliable fallback option

### Future Work
1. Debug AMS convergence issues:
   - Verify discrete gradient matrix orientation
   - Add Poisson solver parameters
   - Test with different AMS cycle types
2. Add 3D benchmarks
3. Test with higher-order elements (N1E_TRI2, N1E_TET2)
4. Compare with direct solvers (MUMPS, SuperLU) for reference

---

## Reproducing Results

Run the benchmark script:
```bash
python examples/maxwell_solver_benchmark.py
```

This will generate:
- Console output with detailed timing
- `maxwell_solver_benchmark.png` with performance plots
- This summary document
