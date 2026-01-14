"""
Performance benchmark for Maxwell equation solvers with different preconditioners.

This script compares:
- HYPRE AMS (specialized for H(curl))
- HYPRE BoomerAMG (general algebraic multigrid)
- ILU (incomplete LU factorization)
- Jacobi (diagonal preconditioner)

Metrics:
- Iteration count
- Solve time
- Setup time
- Residual norm
"""

import numpy as np
import time
from jax_fem.generate_mesh import Mesh
from jax_fem.hcurl_fe import (
    HCurlFiniteElement, 
    compute_curl_curl_matrix, 
    compute_mass_matrix,
    compute_discrete_gradient
)
from jax_fem.solver import petsc_solve
from petsc4py import PETSc
import matplotlib.pyplot as plt


def create_test_mesh(nx=8, ny=8):
    """Create unit square mesh for benchmarking."""
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    
    points = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            points.append([x[i], y[j]])
    points = np.array(points)
    
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            
            cells.append([n0, n1, n2])
            cells.append([n0, n2, n3])
    
    cells = np.array(cells, dtype=np.int32)
    return Mesh(points, cells)


def setup_problem(mesh, omega=2.0*np.pi):
    """Setup Maxwell problem."""
    def boundary_location(point):
        x, y = point
        tol = 1e-10
        return (x < tol or x > 1 - tol or y < tol or y > 1 - tol)
    
    def zero_field(point):
        return np.array([0.0, 0.0])
    
    dirichlet_bc_info = [[boundary_location], [zero_field]]
    
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=2,
        ele_type='N1E_TRI1',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    
    K = compute_curl_curl_matrix(fe)
    M = compute_mass_matrix(fe)
    A_scipy = K + omega**2 * M
    
    # RHS with simple source
    b = np.ones(fe.num_total_edges) * 0.1
    if fe.edge_inds_list:
        for edge_inds, vals in zip(fe.edge_inds_list, fe.vals_list):
            b[edge_inds] = vals
    
    G = compute_discrete_gradient(fe)
    
    return fe, A_scipy, b, G


def benchmark_solver(A_scipy, b, fe, G, solver_config, name):
    """Benchmark a single solver configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    # Convert to PETSc
    A_petsc = PETSc.Mat().createAIJ(
        size=A_scipy.shape,
        csr=(
            A_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_scipy.indices.astype(PETSc.IntType, copy=False),
            A_scipy.data
        )
    )
    
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            A_petsc.zeroRows(edge_inds.astype(np.int32))
    
    # Setup solver
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), b)
    
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(solver_config['ksp_type'])
    
    pc = ksp.getPC()
    pc.setType(solver_config['pc_type'])
    
    # Configure preconditioner
    if solver_config['pc_type'] == 'hypre':
        pc.setHYPREType(solver_config['hypre_type'])
        
        if solver_config['hypre_type'] == 'ams':
            G_petsc = PETSc.Mat().createAIJ(
                size=G.shape,
                csr=(
                    G.indptr.astype(PETSc.IntType, copy=False),
                    G.indices.astype(PETSc.IntType, copy=False),
                    G.data
                )
            )
            pc.setHYPREDiscreteGradient(G_petsc)
            pc.setCoordinates(fe.points)
    
    ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
    ksp.setFromOptions()
    
    # Measure setup time
    setup_start = time.time()
    ksp.setUp()
    setup_time = time.time() - setup_start
    
    # Solve
    x = PETSc.Vec().createSeq(len(b))
    
    solve_start = time.time()
    ksp.solve(rhs, x)
    solve_time = time.time() - solve_start
    
    # Get metrics
    iterations = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    
    # Compute residual
    y = PETSc.Vec().createSeq(len(b))
    A_petsc.mult(x, y)
    residual = np.linalg.norm(y.getArray() - rhs.getArray())
    
    results = {
        'name': name,
        'iterations': iterations,
        'setup_time': setup_time,
        'solve_time': solve_time,
        'total_time': setup_time + solve_time,
        'residual': residual,
        'converged': reason > 0,
        'reason': reason
    }
    
    print(f"  Iterations:   {iterations}")
    print(f"  Setup time:   {setup_time:.4f} s")
    print(f"  Solve time:   {solve_time:.4f} s")
    print(f"  Total time:   {results['total_time']:.4f} s")
    print(f"  Residual:     {residual:.6e}")
    print(f"  Converged:    {results['converged']} (reason={reason})")
    
    return results


def run_benchmark_suite(mesh_sizes=[4, 8, 12, 16]):
    """Run benchmark suite with different mesh sizes."""
    all_results = []
    
    solver_configs = [
        {
            'name': 'HYPRE AMS',
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'hypre_type': 'ams'
        },
        {
            'name': 'HYPRE BoomerAMG',
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'hypre_type': 'boomeramg'
        },
        {
            'name': 'ILU',
            'ksp_type': 'gmres',
            'pc_type': 'ilu',
        },
        {
            'name': 'Jacobi',
            'ksp_type': 'cg',
            'pc_type': 'jacobi',
        },
    ]
    
    for nx in mesh_sizes:
        print(f"\n{'#'*70}")
        print(f"# Mesh size: {nx} x {nx} ({2*nx*nx} triangles)")
        print(f"{'#'*70}")
        
        mesh = create_test_mesh(nx=nx, ny=nx)
        fe, A_scipy, b, G = setup_problem(mesh)
        
        print(f"  DOFs: {fe.num_total_edges}")
        print(f"  Matrix size: {A_scipy.shape}")
        print(f"  Matrix nnz: {A_scipy.nnz}")
        
        for config in solver_configs:
            try:
                result = benchmark_solver(A_scipy, b, fe, G, config, config['name'])
                result['mesh_size'] = nx
                result['num_dofs'] = fe.num_total_edges
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    'name': config['name'],
                    'mesh_size': nx,
                    'num_dofs': fe.num_total_edges,
                    'iterations': -1,
                    'setup_time': 0,
                    'solve_time': 0,
                    'total_time': 0,
                    'residual': np.inf,
                    'converged': False,
                    'reason': -999
                })
    
    return all_results


def visualize_results(results):
    """Create visualization of benchmark results."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Filter successful runs
    df_success = df[df['converged'] == True]
    
    if len(df_success) == 0:
        print("No successful runs to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Iterations vs DOFs
    ax = axes[0, 0]
    for name in df_success['name'].unique():
        data = df_success[df_success['name'] == name]
        ax.plot(data['num_dofs'], data['iterations'], 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Number of DOFs')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Iteration Count vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Solve time vs DOFs
    ax = axes[0, 1]
    for name in df_success['name'].unique():
        data = df_success[df_success['name'] == name]
        ax.plot(data['num_dofs'], data['solve_time'], 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Number of DOFs')
    ax.set_ylabel('Solve Time (s)')
    ax.set_title('Solve Time vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 3: Total time vs DOFs
    ax = axes[1, 0]
    for name in df_success['name'].unique():
        data = df_success[df_success['name'] == name]
        ax.plot(data['num_dofs'], data['total_time'], 'o-', label=name, linewidth=2, markersize=8)
    ax.set_xlabel('Number of DOFs')
    ax.set_ylabel('Total Time (setup + solve, s)')
    ax.set_title('Total Time vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 4: Speedup comparison (relative to Jacobi)
    ax = axes[1, 1]
    
    jacobi_data = df_success[df_success['name'] == 'Jacobi']
    
    for name in df_success['name'].unique():
        if name == 'Jacobi':
            continue
        
        data = df_success[df_success['name'] == name]
        speedups = []
        dofs = []
        
        for _, row in data.iterrows():
            jacobi_row = jacobi_data[jacobi_data['mesh_size'] == row['mesh_size']]
            if len(jacobi_row) > 0:
                speedup = jacobi_row.iloc[0]['total_time'] / row['total_time']
                speedups.append(speedup)
                dofs.append(row['num_dofs'])
        
        if speedups:
            ax.plot(dofs, speedups, 'o-', label=name, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline (Jacobi)')
    ax.set_xlabel('Number of DOFs')
    ax.set_ylabel('Speedup vs Jacobi')
    ax.set_title('Speedup Comparison (Total Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('/mnt/d/pythoncode/jax-fem/jax-fem-main/examples/maxwell_solver_benchmark.png', dpi=150)
    print(f"\nBenchmark plot saved to: maxwell_solver_benchmark.png")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(df_success.to_string(index=False))


def main():
    print("=" * 70)
    print("Maxwell Solver Benchmark: AMS vs ILU vs BoomerAMG vs Jacobi")
    print("=" * 70)
    
    # Run benchmarks with increasing mesh sizes
    mesh_sizes = [4, 8, 12, 16]
    results = run_benchmark_suite(mesh_sizes)
    
    # Visualize
    visualize_results(results)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
