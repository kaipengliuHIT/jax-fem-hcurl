"""
Waveguide example with complex boundary conditions using H(curl) Nedelec elements.

This example solves a 2D waveguide problem with:
- Perfect Electric Conductor (PEC) on side walls: E x n = 0
- Incident wave on left boundary: E x n = E_inc x n
- Absorbing boundary condition on right boundary (simplified)

Problem:
    curl curl E - omega^2 E = 0    in waveguide
    E x n = E_inc x n              on left (input)
    E x n = 0                      on top/bottom (PEC walls)
    (simplified ABC)               on right (output)

This demonstrates:
- Mixed boundary conditions (PEC + incident wave)
- Non-homogeneous Dirichlet BC
- Waveguide mode propagation
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


def create_waveguide_mesh(Lx=2.0, Ly=1.0, nx=20, ny=10):
    """Create rectangular waveguide mesh [0,Lx] x [0,Ly]."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    
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
    return Mesh(points, cells), Lx, Ly


def pec_walls_location(point, Ly):
    """PEC on top and bottom walls."""
    x, y = point
    tol = 1e-10
    return (y < tol or y > Ly - tol)


def input_port_location(point, Lx, Ly):
    """Input port on left boundary."""
    x, y = point
    tol = 1e-10
    return (x < tol)


def output_port_location(point, Lx, Ly):
    """Output port on right boundary (simplified ABC)."""
    x, y = point
    tol = 1e-10
    return (x > Lx - tol)


def incident_wave_field(point, omega, Ly):
    """
    Incident TE10 mode: E_z = sin(pi*y/Ly) * exp(i*beta*x)
    For 2D H(curl) with E = [E_x, E_y], we use E_y component.
    
    At x=0 (input), we set the tangential component.
    """
    x, y = point
    # TE10 mode cutoff: beta = sqrt(omega^2 - (pi/Ly)^2)
    k_c = np.pi / Ly
    beta = np.sqrt(max(omega**2 - k_c**2, 0))
    
    # For 2D, E field in z-direction (out of plane)
    # But we're solving for in-plane E, so we use a simplified excitation
    E_amplitude = 1.0
    return np.array([0.0, E_amplitude * np.sin(np.pi * y / Ly)])


def zero_field(point):
    """Zero field for PEC walls."""
    return np.array([0.0, 0.0])


def main():
    print("=" * 70)
    print("Waveguide Example with Mixed Boundary Conditions")
    print("=" * 70)
    
    # Parameters
    Lx, Ly = 2.0, 1.0  # Waveguide dimensions
    omega = 2.0 * np.pi  # Frequency
    
    # 1. Create mesh
    print("\n[1/6] Creating waveguide mesh...")
    mesh, Lx, Ly = create_waveguide_mesh(Lx=Lx, Ly=Ly, nx=30, ny=15)
    print(f"  Waveguide: {Lx} x {Ly}")
    print(f"  Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} triangles")
    
    # 2. Setup H(curl) finite element space with mixed BC
    print("\n[2/6] Setting up H(curl) space with mixed BC...")
    
    # Define boundary conditions
    # PEC walls (top/bottom): E x n = 0
    pec_loc = lambda pt: pec_walls_location(pt, Ly)
    pec_val = zero_field
    
    # Input port (left): E x n = E_inc x n
    input_loc = lambda pt: input_port_location(pt, Lx, Ly)
    input_val = lambda pt: incident_wave_field(pt, omega, Ly)
    
    # Combine boundary conditions
    dirichlet_bc_info = [
        [pec_loc, input_loc],
        [pec_val, input_val]
    ]
    
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=2,
        ele_type='N1E_TRI1',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info
    )
    print(f"  DOFs: {fe.num_total_edges} edges")
    print(f"  BC edges: {sum(len(inds) for inds in fe.edge_inds_list)}")
    
    # 3. Assemble system matrices
    print("\n[3/6] Assembling matrices...")
    K = compute_curl_curl_matrix(fe)
    M = compute_mass_matrix(fe)
    
    # Add small damping for absorbing BC (simplified)
    damping = 0.1
    A_scipy = K + (omega**2 - 1j * omega * damping) * M
    print(f"  System matrix: {A_scipy.shape}, nnz={A_scipy.nnz}")
    print(f"  Matrix is complex: {np.iscomplexobj(A_scipy.data)}")
    
    # 4. Compute discrete gradient
    print("\n[4/6] Computing discrete gradient for AMS...")
    G = compute_discrete_gradient(fe)
    
    # 5. Assemble RHS (zero for homogeneous Helmholtz)
    print("\n[5/6] Assembling RHS...")
    b = np.zeros(fe.num_total_edges, dtype=np.complex128)
    
    # Apply boundary conditions
    if fe.edge_inds_list:
        for edge_inds, vals in zip(fe.edge_inds_list, fe.vals_list):
            b[edge_inds] = vals
    
    print(f"  RHS norm: {np.linalg.norm(b):.6e}")
    
    # 6. Solve
    print("\n[6/6] Solving waveguide problem...")
    
    # For complex problems, we need to handle real/imag parts separately
    # or use a complex-capable solver
    print("  Note: Using real part only for this demonstration")
    A_real = A_scipy.real
    b_real = b.real
    
    A_petsc = PETSc.Mat().createAIJ(
        size=A_real.shape,
        csr=(
            A_real.indptr.astype(PETSc.IntType, copy=False),
            A_real.indices.astype(PETSc.IntType, copy=False),
            A_real.data
        )
    )
    
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            A_petsc.zeroRows(edge_inds.astype(np.int32))
    
    solver_options = {
        'petsc_solver': {
            'ksp_type': 'gmres',  # Use GMRES for non-symmetric problems
            'pc_type': 'hypre',
            'hypre_type': 'ams',
            'discrete_gradient': G,
            'coordinates': mesh.points,
        }
    }
    
    try:
        x = petsc_solve(
            A_petsc, 
            b_real, 
            ksp_type=solver_options['petsc_solver']['ksp_type'],
            pc_type=solver_options['petsc_solver']['pc_type'],
            pc_options=solver_options['petsc_solver']
        )
        
        print(f"\n{'='*70}")
        print("WAVEGUIDE SOLUTION CONVERGED!")
        print(f"{'='*70}")
        print(f"  Solution norm: {np.linalg.norm(x):.6e}")
        print(f"  Max value: {np.max(np.abs(x)):.6e}")
        
        residual = A_real @ x - b_real
        print(f"  Residual norm: {np.linalg.norm(residual):.6e}")
        
        # 7. Visualize
        print("\n[7/7] Visualizing waveguide solution...")
        visualize_waveguide(fe, x, mesh, Lx, Ly)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}")
        raise


def visualize_waveguide(fe, x, mesh, Lx, Ly):
    """Visualize waveguide field distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Edge DOF values
    ax = axes[0, 0]
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
        s=10,
        alpha=0.6
    )
    ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells, 'k-', alpha=0.1, linewidth=0.3)
    ax.set_xlabel('x (propagation direction)')
    ax.set_ylabel('y (transverse)')
    ax.set_title('Edge DOF Values')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2: Field magnitude
    ax = axes[0, 1]
    
    # Create regular grid for interpolation
    nx_plot, ny_plot = 50, 25
    x_plot = np.linspace(0, Lx, nx_plot)
    y_plot = np.linspace(0, Ly, ny_plot)
    X, Y = np.meshgrid(x_plot, y_plot)
    
    # Interpolate field magnitude
    field_mag = np.zeros((ny_plot, nx_plot))
    
    for i in range(nx_plot):
        for j in range(ny_plot):
            pt = np.array([x_plot[i], y_plot[j]])
            
            # Find containing cell (simple search)
            for cell_idx in range(fe.num_cells):
                cell_nodes = mesh.cells[cell_idx]
                cell_pts = mesh.points[cell_nodes]
                
                # Check if point is in triangle (barycentric coordinates)
                v0 = cell_pts[2] - cell_pts[0]
                v1 = cell_pts[1] - cell_pts[0]
                v2 = pt - cell_pts[0]
                
                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot02 = np.dot(v0, v2)
                dot11 = np.dot(v1, v1)
                dot12 = np.dot(v1, v2)
                
                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                
                if (u >= -0.01) and (v >= -0.01) and (u + v <= 1.01):
                    # Point is in this cell
                    cell_dofs = fe.cell_to_dofs[cell_idx]
                    field = np.zeros(2)
                    
                    q = 0  # Use first quad point as approximation
                    for k, dof_k in enumerate(cell_dofs):
                        field += x[dof_k] * fe.shape_vals_physical[cell_idx, q, k, :]
                    
                    field_mag[j, i] = np.linalg.norm(field)
                    break
    
    contour = ax.contourf(X, Y, field_mag, levels=20, cmap='viridis')
    ax.set_xlabel('x (propagation direction)')
    ax.set_ylabel('y (transverse)')
    ax.set_title('Field Magnitude |E|')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    plt.colorbar(contour, ax=ax)
    
    # Plot 3: Field along centerline
    ax = axes[1, 0]
    
    centerline_x = []
    centerline_mag = []
    
    for cell_idx in range(fe.num_cells):
        center = np.mean(mesh.points[mesh.cells[cell_idx]], axis=0)
        
        if abs(center[1] - Ly/2) < 0.05:  # Near centerline
            cell_dofs = fe.cell_to_dofs[cell_idx]
            field = np.zeros(2)
            
            q = 0
            for i, dof_i in enumerate(cell_dofs):
                field += x[dof_i] * fe.shape_vals_physical[cell_idx, q, i, :]
            
            centerline_x.append(center[0])
            centerline_mag.append(np.linalg.norm(field))
    
    if centerline_x:
        idx_sort = np.argsort(centerline_x)
        centerline_x = np.array(centerline_x)[idx_sort]
        centerline_mag = np.array(centerline_mag)[idx_sort]
        
        ax.plot(centerline_x, centerline_mag, 'b-', linewidth=2)
        ax.set_xlabel('x (propagation direction)')
        ax.set_ylabel('|E|')
        ax.set_title('Field Magnitude along Centerline (y=Ly/2)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Boundary conditions visualization
    ax = axes[1, 1]
    
    # Mark different boundary regions
    for edge_idx, (n0, n1) in enumerate(fe.edges):
        p0 = mesh.points[n0]
        p1 = mesh.points[n1]
        center = 0.5 * (p0 + p1)
        
        tol = 1e-10
        if center[1] < tol or center[1] > Ly - tol:
            # PEC walls
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2, alpha=0.5)
        elif center[0] < tol:
            # Input port
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'g-', linewidth=2, alpha=0.5)
        elif center[0] > Lx - tol:
            # Output port
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Boundary Conditions')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.legend(['PEC (top/bottom)', 'Input (left)', 'Output (right)'], loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/mnt/d/pythoncode/jax-fem/jax-fem-main/examples/maxwell_waveguide_solution.png', dpi=150)
    print(f"  Waveguide solution plot saved to: maxwell_waveguide_solution.png")
    plt.close()


if __name__ == '__main__':
    main()
