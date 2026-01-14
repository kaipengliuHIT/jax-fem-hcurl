"""
3D Maxwell equation example using H(curl) Nedelec tetrahedral elements with HYPRE AMS.

This example solves a 3D time-harmonic Maxwell problem:
    curl curl E - omega^2 E = f    in Omega
    E x n = 0                       on boundary

where E is the electric field (3D vector), omega is frequency, and f is a source term.

This demonstrates:
- HCurlFiniteElement with 3D Nedelec edge elements (N1E_TET1)
- Assembly of curl-curl and mass matrices in 3D
- Discrete gradient matrix for AMS
- PETSc solver with HYPRE AMS preconditioner
- 3D visualization
"""

import numpy as np
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.hcurl_fe import (
    HCurlFiniteElement, 
    compute_curl_curl_matrix, 
    compute_mass_matrix,
    compute_discrete_gradient
)
from jax_fem.solver import petsc_solve
from petsc4py import PETSc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_unit_cube_mesh(nx=3, ny=3, nz=3):
    """Create a tetrahedral mesh on unit cube [0,1]^3."""
    # Create simple tetrahedral mesh manually
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    z = np.linspace(0, 1, nz + 1)
    
    points = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append([x[i], y[j], z[k]])
    points = np.array(points)
    
    # Split each cube into 5 tetrahedra
    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Node indices for cube (i,j,k)
                n0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nx + 1)
                n3 = n2 + 1
                n4 = n0 + (nx + 1) * (ny + 1)
                n5 = n4 + 1
                n6 = n4 + (nx + 1)
                n7 = n6 + 1
                
                # Split cube into 5 tetrahedra (standard subdivision)
                cells.append([n0, n1, n2, n5])
                cells.append([n2, n3, n1, n7])
                cells.append([n5, n2, n7, n1])
                cells.append([n5, n7, n4, n2])
                cells.append([n4, n2, n6, n7])
    
    cells = np.array(cells, dtype=np.int32)
    return Mesh(points, cells)


def boundary_location(point):
    """Identify boundary points (all 6 faces of unit cube)."""
    x, y, z = point
    tol = 1e-10
    return (x < tol or x > 1 - tol or 
            y < tol or y > 1 - tol or 
            z < tol or z > 1 - tol)


def zero_field(point):
    """Zero boundary condition for E x n = 0."""
    return np.array([0.0, 0.0, 0.0])


def source_term(point):
    """Source term f(x,y,z) = [sin(pi*x)*sin(pi*y)*sin(pi*z), 0, 0]."""
    x, y, z = point
    return np.array([
        np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
        0.0,
        0.0
    ])


def main():
    print("=" * 70)
    print("3D Maxwell Equation with H(curl) Nedelec Elements + HYPRE AMS")
    print("=" * 70)
    
    # 1. Create mesh
    print("\n[1/6] Creating 3D tetrahedral mesh...")
    mesh = create_unit_cube_mesh(nx=4, ny=4, nz=4)
    print(f"  Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} tetrahedra")
    
    # 2. Setup H(curl) finite element space
    print("\n[2/6] Setting up 3D H(curl) finite element space...")
    dirichlet_bc_info = [[boundary_location], [zero_field]]
    
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=3,
        ele_type='N1E_TET1',  # Nedelec first kind, degree 1, tetrahedron
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    print(f"  DOFs: {fe.num_total_edges} edges (3D H(curl) space)")
    print(f"  Element: {fe.ele_type}, {fe.num_quads} quad points per cell")
    
    # 3. Assemble system matrices
    print("\n[3/6] Assembling 3D curl-curl and mass matrices...")
    K = compute_curl_curl_matrix(fe)  # curl-curl stiffness
    M = compute_mass_matrix(fe)       # mass matrix
    
    omega = 2.0 * np.pi  # frequency
    A_scipy = K + omega**2 * M
    print(f"  System matrix: {A_scipy.shape}, nnz={A_scipy.nnz}")
    
    # 4. Compute discrete gradient for AMS
    print("\n[4/6] Computing discrete gradient matrix G for 3D AMS...")
    G = compute_discrete_gradient(fe)
    print(f"  Discrete gradient G: {G.shape} (edges x nodes)")
    print(f"  G maps H1 nodal space -> H(curl) edge space")
    
    # 5. Assemble right-hand side
    print("\n[5/6] Assembling right-hand side...")
    b = np.zeros(fe.num_total_edges)
    
    # Project source term onto edge basis
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
    print("\n[6/6] Solving 3D problem with PETSc + HYPRE AMS preconditioner...")
    
    # Convert to PETSc format
    A_petsc = PETSc.Mat().createAIJ(
        size=A_scipy.shape,
        csr=(
            A_scipy.indptr.astype(PETSc.IntType, copy=False),
            A_scipy.indices.astype(PETSc.IntType, copy=False),
            A_scipy.data
        )
    )
    
    # Apply boundary conditions to matrix
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
            'coordinates': mesh.points,
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
        print("3D SOLUTION CONVERGED!")
        print(f"{'='*70}")
        print(f"  Solution norm: {np.linalg.norm(x):.6e}")
        print(f"  Max value: {np.max(np.abs(x)):.6e}")
        
        # Verify solution
        residual = A_scipy @ x - b
        print(f"  Residual norm: {np.linalg.norm(residual):.6e}")
        
        # 7. Visualize
        print("\n[7/7] Visualizing 3D solution...")
        visualize_3d_solution(fe, x, mesh)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}")
        raise


def visualize_3d_solution(fe, x, mesh):
    """Visualize the 3D edge-based solution."""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Edge DOF values (scatter plot of edge centers)
    ax1 = fig.add_subplot(131, projection='3d')
    edge_centers = []
    edge_values = []
    
    for edge_idx, (n0, n1) in enumerate(fe.edges):
        center = 0.5 * (mesh.points[n0] + mesh.points[n1])
        edge_centers.append(center)
        edge_values.append(x[edge_idx])
    
    edge_centers = np.array(edge_centers)
    edge_values = np.array(edge_values)
    
    scatter = ax1.scatter(
        edge_centers[:, 0], 
        edge_centers[:, 1], 
        edge_centers[:, 2],
        c=edge_values, 
        cmap='RdBu_r',
        s=20,
        alpha=0.6
    )
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Edge DOF Values')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # Plot 2: Field magnitude at cell centers
    ax2 = fig.add_subplot(132, projection='3d')
    cell_centers = []
    field_magnitudes = []
    
    for cell_idx in range(min(fe.num_cells, 500)):  # Limit for visualization
        cell_dofs = fe.cell_to_dofs[cell_idx]
        field = np.zeros(3)
        
        q = 0  # Use first quadrature point
        for i, dof_i in enumerate(cell_dofs):
            field += x[dof_i] * fe.shape_vals_physical[cell_idx, q, i, :]
        
        center = np.mean(mesh.points[mesh.cells[cell_idx]], axis=0)
        cell_centers.append(center)
        field_magnitudes.append(np.linalg.norm(field))
    
    cell_centers = np.array(cell_centers)
    field_magnitudes = np.array(field_magnitudes)
    
    scatter = ax2.scatter(
        cell_centers[:, 0], 
        cell_centers[:, 1], 
        cell_centers[:, 2],
        c=field_magnitudes, 
        cmap='viridis',
        s=30,
        alpha=0.6
    )
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Field Magnitude |E|')
    plt.colorbar(scatter, ax=ax2, shrink=0.5)
    
    # Plot 3: Slice through z=0.5
    ax3 = fig.add_subplot(133)
    
    # Find cells near z=0.5 plane
    slice_cells = []
    slice_magnitudes = []
    slice_centers_2d = []
    
    for cell_idx in range(fe.num_cells):
        center = np.mean(mesh.points[mesh.cells[cell_idx]], axis=0)
        if abs(center[2] - 0.5) < 0.15:  # Tolerance for slice
            cell_dofs = fe.cell_to_dofs[cell_idx]
            field = np.zeros(3)
            
            q = 0
            for i, dof_i in enumerate(cell_dofs):
                field += x[dof_i] * fe.shape_vals_physical[cell_idx, q, i, :]
            
            slice_cells.append(cell_idx)
            slice_magnitudes.append(np.linalg.norm(field))
            slice_centers_2d.append([center[0], center[1]])
    
    if slice_centers_2d:
        slice_centers_2d = np.array(slice_centers_2d)
        slice_magnitudes = np.array(slice_magnitudes)
        
        scatter = ax3.scatter(
            slice_centers_2d[:, 0],
            slice_centers_2d[:, 1],
            c=slice_magnitudes,
            cmap='viridis',
            s=100
        )
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Field Magnitude |E| (z≈0.5 slice)')
        ax3.set_aspect('equal')
        plt.colorbar(scatter, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('/mnt/d/pythoncode/jax-fem/jax-fem-main/examples/maxwell_3d_ams_solution.png', dpi=150)
    print(f"  3D solution plot saved to: maxwell_3d_ams_solution.png")
    plt.close()


if __name__ == '__main__':
    main()
