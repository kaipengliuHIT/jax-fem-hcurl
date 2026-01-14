"""
Debug script for HYPRE AMS convergence issues.

This script analyzes potential causes of AMS convergence failure:
1. Discrete gradient matrix G properties
2. Coordinate vector setup
3. AMS parameter configuration
4. Matrix conditioning
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from jax_fem.generate_mesh import Mesh
from jax_fem.hcurl_fe import (
    HCurlFiniteElement, 
    compute_curl_curl_matrix, 
    compute_mass_matrix,
    compute_discrete_gradient
)
from petsc4py import PETSc


def create_test_mesh(nx=8, ny=8):
    """Create unit square mesh."""
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


def analyze_discrete_gradient(G, fe):
    """Analyze discrete gradient matrix properties."""
    print("\n" + "="*70)
    print("DISCRETE GRADIENT MATRIX ANALYSIS")
    print("="*70)
    
    print(f"\nShape: {G.shape} (edges x nodes)")
    print(f"Non-zeros: {G.nnz}")
    print(f"Density: {G.nnz / (G.shape[0] * G.shape[1]):.4f}")
    
    # Check row sums (should be 0 for each edge: -1 + 1 = 0)
    row_sums = np.array(G.sum(axis=1)).flatten()
    print(f"\nRow sums (should be 0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    
    # Check column sums
    col_sums = np.array(G.sum(axis=0)).flatten()
    print(f"Column sums: min={col_sums.min():.2f}, max={col_sums.max():.2f}")
    
    # Check entries per row (should be exactly 2)
    entries_per_row = np.diff(G.indptr)
    print(f"Entries per row: min={entries_per_row.min()}, max={entries_per_row.max()}")
    
    # Check values (should be -1 and +1)
    unique_vals = np.unique(G.data)
    print(f"Unique values: {unique_vals}")
    
    # Check G^T G (should be graph Laplacian)
    GtG = G.T @ G
    print(f"\nG^T @ G shape: {GtG.shape}")
    print(f"G^T @ G diagonal: min={GtG.diagonal().min():.2f}, max={GtG.diagonal().max():.2f}")
    
    # Check kernel of G^T (should be constant vector)
    # For proper G, G @ ones = 0
    ones = np.ones(G.shape[1])
    G_ones = G @ ones
    print(f"\nG @ ones (should be 0): norm = {np.linalg.norm(G_ones):.6e}")
    
    return True


def analyze_system_matrix(A, K, M, omega):
    """Analyze system matrix properties."""
    print("\n" + "="*70)
    print("SYSTEM MATRIX ANALYSIS")
    print("="*70)
    
    print(f"\nCurl-curl matrix K:")
    print(f"  Shape: {K.shape}")
    print(f"  Symmetric: {np.allclose(K.data, K.T.data)}")
    print(f"  Frobenius norm: {sparse_norm(K):.6e}")
    
    print(f"\nMass matrix M:")
    print(f"  Shape: {M.shape}")
    print(f"  Symmetric: {np.allclose(M.data, M.T.data)}")
    print(f"  Frobenius norm: {sparse_norm(M):.6e}")
    
    print(f"\nSystem matrix A = K + omega^2 * M:")
    print(f"  omega = {omega:.4f}")
    print(f"  omega^2 = {omega**2:.4f}")
    print(f"  Shape: {A.shape}")
    print(f"  Frobenius norm: {sparse_norm(A):.6e}")
    
    # Check diagonal dominance
    diag = np.abs(A.diagonal())
    off_diag_sum = np.array(np.abs(A).sum(axis=1)).flatten() - diag
    diag_dominant = np.sum(diag >= off_diag_sum) / len(diag)
    print(f"  Diagonal dominance ratio: {diag_dominant:.2%}")
    
    # Estimate condition number (expensive for large matrices)
    if A.shape[0] <= 500:
        try:
            from scipy.sparse.linalg import eigsh
            eig_max = eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
            eig_min = eigsh(A, k=1, which='SM', return_eigenvectors=False)[0]
            print(f"  Eigenvalue range: [{eig_min:.6e}, {eig_max:.6e}]")
            print(f"  Condition number estimate: {abs(eig_max/eig_min):.6e}")
        except:
            print("  Could not compute eigenvalues")
    
    return True


def test_ams_with_different_configs(A_scipy, b, G, coords, fe):
    """Test AMS with different configurations."""
    print("\n" + "="*70)
    print("AMS CONFIGURATION TESTS")
    print("="*70)
    
    # Convert matrices to PETSc
    A_petsc = PETSc.Mat().createAIJ(
        size=A_scipy.shape,
        csr=(
            A_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_scipy.indices.astype(PETSc.IntType, copy=False),
            A_scipy.data
        )
    )
    
    G_petsc = PETSc.Mat().createAIJ(
        size=G.shape,
        csr=(
            G.indptr.astype(PETSc.IntType, copy=False),
            G.indices.astype(PETSc.IntType, copy=False),
            G.data
        )
    )
    
    # Apply BC
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            A_petsc.zeroRows(edge_inds.astype(np.int32))
    
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), b)
    
    configs = [
        {
            'name': 'AMS default',
            'hypre_type': 'ams',
            'use_coords': True,
            'ams_cycle': None,
        },
        {
            'name': 'AMS cycle=1',
            'hypre_type': 'ams',
            'use_coords': True,
            'ams_cycle': 1,
        },
        {
            'name': 'AMS cycle=13',
            'hypre_type': 'ams',
            'use_coords': True,
            'ams_cycle': 13,
        },
        {
            'name': 'AMS no coords',
            'hypre_type': 'ams',
            'use_coords': False,
            'ams_cycle': None,
        },
        {
            'name': 'BoomerAMG (reference)',
            'hypre_type': 'boomeramg',
            'use_coords': False,
            'ams_cycle': None,
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        try:
            ksp = PETSc.KSP().create()
            ksp.setOperators(A_petsc)
            ksp.setType('cg')
            
            pc = ksp.getPC()
            pc.setType('hypre')
            pc.setHYPREType(config['hypre_type'])
            
            if config['hypre_type'] == 'ams':
                pc.setHYPREDiscreteGradient(G_petsc)
                
                if config['use_coords']:
                    pc.setCoordinates(coords)
                
                # Set AMS options via PETSc options
                if config['ams_cycle'] is not None:
                    PETSc.Options().setValue('-pc_hypre_ams_cycle_type', str(config['ams_cycle']))
            
            ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)
            ksp.setFromOptions()
            
            x = PETSc.Vec().createSeq(len(b))
            ksp.solve(rhs, x)
            
            iterations = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            
            # Compute residual
            y = PETSc.Vec().createSeq(len(b))
            A_petsc.mult(x, y)
            residual = np.linalg.norm(y.getArray() - rhs.getArray())
            
            converged = reason > 0
            
            print(f"  Iterations: {iterations}")
            print(f"  Converged: {converged} (reason={reason})")
            print(f"  Residual: {residual:.6e}")
            
            results.append({
                'name': config['name'],
                'iterations': iterations,
                'converged': converged,
                'reason': reason,
                'residual': residual
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': config['name'],
                'iterations': -1,
                'converged': False,
                'reason': -999,
                'residual': np.inf
            })
    
    return results


def check_edge_orientation_consistency(fe):
    """Check if edge orientations are consistent."""
    print("\n" + "="*70)
    print("EDGE ORIENTATION ANALYSIS")
    print("="*70)
    
    # For each cell, check edge orientations
    edge_to_cells = {}
    for cell_idx in range(fe.num_cells):
        cell_dofs = fe.cell_to_dofs[cell_idx]
        for dof in cell_dofs:
            if dof not in edge_to_cells:
                edge_to_cells[dof] = []
            edge_to_cells[dof].append(cell_idx)
    
    # Count edges shared by multiple cells
    shared_edges = sum(1 for e, cells in edge_to_cells.items() if len(cells) > 1)
    boundary_edges = sum(1 for e, cells in edge_to_cells.items() if len(cells) == 1)
    
    print(f"Total edges: {fe.num_total_edges}")
    print(f"Interior edges (shared): {shared_edges}")
    print(f"Boundary edges: {boundary_edges}")
    
    # Check edge lengths
    edge_lengths = []
    for n0, n1 in fe.edges:
        length = np.linalg.norm(fe.points[n1] - fe.points[n0])
        edge_lengths.append(length)
    
    edge_lengths = np.array(edge_lengths)
    print(f"\nEdge lengths: min={edge_lengths.min():.6f}, max={edge_lengths.max():.6f}")
    print(f"Edge length ratio: {edge_lengths.max()/edge_lengths.min():.2f}")
    
    return True


def test_scaled_discrete_gradient(A_scipy, b, fe, coords):
    """Test with edge-length scaled discrete gradient."""
    print("\n" + "="*70)
    print("TESTING SCALED DISCRETE GRADIENT")
    print("="*70)
    
    from scipy.sparse import coo_matrix
    
    num_edges = fe.num_total_edges
    num_nodes = fe.num_total_nodes
    
    rows = []
    cols = []
    vals = []
    
    for edge_idx, (n0, n1) in enumerate(fe.edges):
        # Compute edge length
        edge_vec = fe.points[n1] - fe.points[n0]
        edge_length = np.linalg.norm(edge_vec)
        
        # Scale by edge length (this is what HYPRE expects for some formulations)
        rows.extend([edge_idx, edge_idx])
        cols.extend([n0, n1])
        vals.extend([-1.0/edge_length, 1.0/edge_length])
    
    G_scaled = coo_matrix((vals, (rows, cols)), shape=(num_edges, num_nodes)).tocsr()
    
    print(f"Scaled G values: min={min(vals):.6f}, max={max(vals):.6f}")
    
    # Test with scaled G
    A_petsc = PETSc.Mat().createAIJ(
        size=A_scipy.shape,
        csr=(
            A_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_scipy.indices.astype(PETSc.IntType, copy=False),
            A_scipy.data
        )
    )
    
    G_petsc = PETSc.Mat().createAIJ(
        size=G_scaled.shape,
        csr=(
            G_scaled.indptr.astype(PETSc.IntType, copy=False),
            G_scaled.indices.astype(PETSc.IntType, copy=False),
            G_scaled.data
        )
    )
    
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            A_petsc.zeroRows(edge_inds.astype(np.int32))
    
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), b)
    
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType('cg')
    
    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setHYPREType('ams')
    pc.setHYPREDiscreteGradient(G_petsc)
    pc.setCoordinates(coords)
    
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)
    ksp.setFromOptions()
    
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)
    
    iterations = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    
    y = PETSc.Vec().createSeq(len(b))
    A_petsc.mult(x, y)
    residual = np.linalg.norm(y.getArray() - rhs.getArray())
    
    print(f"\nWith scaled G:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {reason > 0} (reason={reason})")
    print(f"  Residual: {residual:.6e}")
    
    return G_scaled, reason > 0


def main():
    print("="*70)
    print("AMS CONVERGENCE DEBUGGING")
    print("="*70)
    
    # Test with problematic mesh size
    for nx in [8, 12]:
        print(f"\n{'#'*70}")
        print(f"# MESH SIZE: {nx} x {nx}")
        print(f"{'#'*70}")
        
        mesh = create_test_mesh(nx=nx, ny=nx)
        
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
        
        print(f"\nMesh: {len(mesh.points)} nodes, {len(mesh.cells)} triangles")
        print(f"DOFs: {fe.num_total_edges} edges")
        
        # Assemble matrices
        omega = 2.0 * np.pi
        K = compute_curl_curl_matrix(fe)
        M = compute_mass_matrix(fe)
        A_scipy = K + omega**2 * M
        
        # RHS
        b = np.ones(fe.num_total_edges) * 0.1
        if fe.edge_inds_list:
            for edge_inds, vals in zip(fe.edge_inds_list, fe.vals_list):
                b[edge_inds] = vals
        
        G = compute_discrete_gradient(fe)
        coords = fe.points.copy()
        
        # Run analyses
        analyze_discrete_gradient(G, fe)
        analyze_system_matrix(A_scipy, K, M, omega)
        check_edge_orientation_consistency(fe)
        
        # Test different AMS configurations
        test_ams_with_different_configs(A_scipy, b, G, coords, fe)
        
        # Test scaled discrete gradient
        G_scaled, success = test_scaled_discrete_gradient(A_scipy, b, fe, coords)
        
        if success:
            print("\n*** SCALED DISCRETE GRADIENT WORKS! ***")
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
