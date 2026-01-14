"""
Minimal Maxwell equation example using H(curl) Nedelec elements with HYPRE AMS preconditioner.

This example solves a 2D time-harmonic Maxwell problem:
    curl curl E - omega^2 E = f    in Omega
    E x n = 0                       on boundary

where E is the electric field (vector), omega is frequency, and f is a source term.

This demonstrates:
- HCurlFiniteElement with Nedelec edge elements (N1E_TRI1)
- Assembly of curl-curl and mass matrices
- Discrete gradient matrix for AMS
- PETSc solver with HYPRE AMS preconditioner
"""

import numpy as np
import jax.numpy as jnp
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


def create_unit_square_mesh(nx=4, ny=4):
    """Create a simple triangular mesh on unit square [0,1]x[0,1]."""
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
            # Node indices for quad (i,j)
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            
            # Split quad into 2 triangles
            cells.append([n0, n1, n2])
            cells.append([n0, n2, n3])
    
    cells = np.array(cells, dtype=np.int32)
    return Mesh(points, cells)


def boundary_location(point):
    """Identify boundary points (all 4 edges of unit square)."""
    x, y = point
    tol = 1e-10
    return (x < tol or x > 1 - tol or y < tol or y > 1 - tol)


def zero_field(point):
    """Zero boundary condition for E x n = 0."""
    return np.array([0.0, 0.0])


def source_term(point):
    """Source term f(x,y) = [sin(pi*x)*sin(pi*y), 0]."""
    x, y = point
    return np.array([np.sin(np.pi * x) * np.sin(np.pi * y), 0.0])


def main():
    print("=" * 70)
    print("Maxwell Equation with H(curl) Nedelec Elements + HYPRE AMS")
    print("=" * 70)
    
    # 1. Create mesh
    print("\n[1/6] Creating mesh...")
    mesh = create_unit_square_mesh(nx=8, ny=8)
    print(f"  Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} triangles")
    
    # 2. Setup H(curl) finite element space
    print("\n[2/6] Setting up H(curl) finite element space...")
    dirichlet_bc_info = [[boundary_location], [zero_field]]
    
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=2,
        ele_type='N1E_TRI1',  # Nedelec first kind, degree 1, triangle
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    print(f"  DOFs: {fe.num_total_edges} edges (H(curl) space)")
    print(f"  Element: {fe.ele_type}, {fe.num_quads} quad points per cell")
    
    # 3. Assemble system matrices
    print("\n[3/6] Assembling curl-curl and mass matrices...")
    K = compute_curl_curl_matrix(fe)  # curl-curl stiffness
    M = compute_mass_matrix(fe)       # mass matrix
    
    omega = 2.0 * np.pi  # frequency
    A_scipy = K + omega**2 * M
    print(f"  System matrix: {A_scipy.shape}, nnz={A_scipy.nnz}")
    
    # 4. Compute discrete gradient for AMS
    print("\n[4/6] Computing discrete gradient matrix G for AMS...")
    G = compute_discrete_gradient(fe)
    print(f"  Discrete gradient G: {G.shape} (edges x nodes)")
    print(f"  G maps H1 nodal space -> H(curl) edge space")
    
    # 5. Assemble right-hand side
    print("\n[5/6] Assembling right-hand side...")
    b = np.zeros(fe.num_total_edges)
    
    # Simple source: project source_term onto edge basis
    # For this minimal example, we use a manufactured RHS
    for cell_idx in range(fe.num_cells):
        cell_dofs = fe.cell_to_dofs[cell_idx]
        quad_points = fe.get_physical_quad_points()[cell_idx]
        
        for q in range(fe.num_quads):
            pt = quad_points[q]
            f_val = source_term(pt)
            
            for i, dof_i in enumerate(cell_dofs):
                # (f, phi_i)
                phi_i = fe.shape_vals_physical[cell_idx, q, i, :]
                b[dof_i] += np.dot(f_val, phi_i) * fe.JxW[cell_idx, q]
    
    # Apply boundary conditions to RHS
    if fe.edge_inds_list:
        for edge_inds, vals in zip(fe.edge_inds_list, fe.vals_list):
            b[edge_inds] = vals
    
    print(f"  RHS norm: {np.linalg.norm(b):.6e}")
    
    # 6. Solve with PETSc + HYPRE AMS
    print("\n[6/6] Solving with PETSc + HYPRE AMS preconditioner...")
    
    # Convert to PETSc format
    A_petsc = PETSc.Mat().createAIJ(
        size=A_scipy.shape,
        csr=(
            A_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_scipy.indices.astype(PETSc.IntType, copy=False),
            A_scipy.data
        )
    )
    
    # Apply boundary conditions to matrix (zero rows)
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            A_petsc.zeroRows(edge_inds.astype(np.int32))
    
    # Solve with AMS
    # Note: Use GMRES instead of CG for better AMS convergence
    solver_options = {
        'petsc_solver': {
            'ksp_type': 'gmres',  # GMRES works better with AMS than CG
            'pc_type': 'hypre',
            'hypre_type': 'ams',
            'discrete_gradient': G,
            'coordinates': mesh.points,  # Node coordinates for AMS
        }
    }
    
    try:
        x = petsc_solve(
            A_petsc, 
            b, 
            ksp_type=solver_options['petsc_solver']['ksp_type'],
            pc_type=solver_options['petsc_solver']['pc_type'],
            pc_options=solver_options['petsc_solver']
        )
        
        print(f"\n{'='*70}")
        print("SOLUTION CONVERGED!")
        print(f"{'='*70}")
        print(f"  Solution norm: {np.linalg.norm(x):.6e}")
        print(f"  Max value: {np.max(np.abs(x)):.6e}")
        
        # Verify solution
        residual = A_scipy @ x - b
        print(f"  Residual norm: {np.linalg.norm(residual):.6e}")
        
        # 7. Visualize (optional)
        print("\n[7/7] Visualizing solution...")
        visualize_solution(fe, x, mesh)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}")
        print("\nPossible causes:")
        print("  1. PETSc not compiled with HYPRE support")
        print("  2. Check: python -c 'from petsc4py import PETSc; print(PETSc.PC.Type.HYPRE)'")
        print("  3. Install HYPRE: conda install -c conda-forge hypre")
        raise


def visualize_solution(fe, x, mesh):
    """Visualize the edge-based solution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Edge DOF values
    ax = axes[0]
    edge_centers = []
    edge_values = []
    
    for edge_idx, (n0, n1) in enumerate(fe.edges):
        center = 0.5 * (mesh.points[n0] + mesh.points[n1])
        edge_centers.append(center)
        edge_values.append(x[edge_idx])
    
    edge_centers = np.array(edge_centers)
    edge_values = np.array(edge_values)
    
    scatter = ax.scatter(
        edge_centers[:, 0], 
        edge_centers[:, 1], 
        c=edge_values, 
        cmap='RdBu_r',
        s=50
    )
    ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells, 'k-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Edge DOF Values')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2: Field magnitude at cell centers
    ax = axes[1]
    cell_centers = []
    field_magnitudes = []
    
    for cell_idx in range(fe.num_cells):
        # Evaluate field at cell center (first quad point as approximation)
        cell_dofs = fe.cell_to_dofs[cell_idx]
        field = np.zeros(2)
        
        q = 0  # Use first quadrature point
        for i, dof_i in enumerate(cell_dofs):
            field += x[dof_i] * fe.shape_vals_physical[cell_idx, q, i, :]
        
        center = np.mean(mesh.points[mesh.cells[cell_idx]], axis=0)
        cell_centers.append(center)
        field_magnitudes.append(np.linalg.norm(field))
    
    cell_centers = np.array(cell_centers)
    field_magnitudes = np.array(field_magnitudes)
    
    scatter = ax.scatter(
        cell_centers[:, 0], 
        cell_centers[:, 1], 
        c=field_magnitudes, 
        cmap='viridis',
        s=100
    )
    ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells, 'k-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Field Magnitude |E|')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig('/mnt/d/pythoncode/jax-fem/jax-fem-main/examples/maxwell_ams_solution.png', dpi=150)
    print(f"  Solution plot saved to: maxwell_ams_solution.png")
    plt.close()


if __name__ == '__main__':
    main()
