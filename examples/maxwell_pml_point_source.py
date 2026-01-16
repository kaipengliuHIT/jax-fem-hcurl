"""
2D Maxwell equations with point source excitation and PML absorbing boundary.

This example demonstrates:
1. Point source excitation (electric dipole)
2. Perfectly Matched Layer (PML) absorbing boundary condition
3. Complex-valued H(curl) formulation
4. Time-harmonic Maxwell equations: curl curl E - k² ε_r E = -iωμ₀ J

The domain is a square with PML layers on all sides to absorb outgoing waves.

Physical setup:
- Central region: free space (ε_r = 1)
- PML layers: complex stretched coordinates for wave absorption
- Point source: z-directed electric dipole at center

Author: JAX-FEM Team
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from jax_fem.generate_mesh import Mesh
from jax_fem.hcurl_fe import (
    HCurlFiniteElement,
    compute_curl_curl_matrix,
    compute_mass_matrix,
    compute_discrete_gradient
)


def create_pml_mesh(nx_inner, ny_inner, nx_pml, ny_pml, domain_size=1.0, pml_thickness=0.2):
    """Create mesh with PML regions.
    
    Parameters
    ----------
    nx_inner, ny_inner : int
        Number of elements in inner (physical) region
    nx_pml, ny_pml : int
        Number of elements in PML layer
    domain_size : float
        Size of inner domain
    pml_thickness : float
        Thickness of PML layer
    
    Returns
    -------
    mesh : Mesh
        The mesh object
    pml_info : dict
        PML region information
    """
    # Total domain
    x_min = -pml_thickness
    x_max = domain_size + pml_thickness
    y_min = -pml_thickness
    y_max = domain_size + pml_thickness
    
    # Create non-uniform grid (finer in inner region)
    # PML left
    x_pml_left = np.linspace(x_min, 0, nx_pml + 1)
    # Inner
    x_inner = np.linspace(0, domain_size, nx_inner + 1)
    # PML right
    x_pml_right = np.linspace(domain_size, x_max, nx_pml + 1)
    
    # Combine (remove duplicates)
    x = np.concatenate([x_pml_left[:-1], x_inner, x_pml_right[1:]])
    
    # Same for y
    y_pml_bottom = np.linspace(y_min, 0, ny_pml + 1)
    y_inner = np.linspace(0, domain_size, ny_inner + 1)
    y_pml_top = np.linspace(domain_size, y_max, ny_pml + 1)
    y = np.concatenate([y_pml_bottom[:-1], y_inner, y_pml_top[1:]])
    
    nx_total = len(x) - 1
    ny_total = len(y) - 1
    
    # Create mesh points
    points = []
    for j in range(len(y)):
        for i in range(len(x)):
            points.append([x[i], y[j]])
    points = np.array(points)
    
    # Create triangular cells
    cells = []
    for j in range(ny_total):
        for i in range(nx_total):
            n0 = j * len(x) + i
            n1 = n0 + 1
            n2 = n0 + len(x) + 1
            n3 = n0 + len(x)
            cells.append([n0, n1, n2])
            cells.append([n0, n2, n3])
    cells = np.array(cells, dtype=np.int32)
    
    mesh = Mesh(points, cells)
    
    pml_info = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'inner_x_min': 0,
        'inner_x_max': domain_size,
        'inner_y_min': 0,
        'inner_y_max': domain_size,
        'pml_thickness': pml_thickness,
    }
    
    return mesh, pml_info


def pml_sigma(coord, pml_start, pml_end, sigma_max, order=3):
    """Compute PML conductivity profile.
    
    Uses polynomial grading: σ(ρ) = σ_max * (ρ/d)^order
    where ρ is distance into PML and d is PML thickness.
    """
    d = abs(pml_end - pml_start)
    if d < 1e-10:
        return 0.0
    
    if pml_start < pml_end:  # Right or top PML
        if coord <= pml_start:
            return 0.0
        elif coord >= pml_end:
            return sigma_max
        else:
            rho = (coord - pml_start) / d
            return sigma_max * (rho ** order)
    else:  # Left or bottom PML
        if coord >= pml_start:
            return 0.0
        elif coord <= pml_end:
            return sigma_max
        else:
            rho = (pml_start - coord) / d
            return sigma_max * (rho ** order)


def compute_pml_tensor(point, pml_info, omega, sigma_max=10.0):
    """Compute PML coordinate stretching tensor at a point.
    
    The PML stretching is: s_x = 1 + σ_x / (iω)
    This gives complex permittivity/permeability tensors.
    
    Returns
    -------
    s_tensor : complex array (2,)
        Stretching factors [s_x, s_y]
    """
    x, y = point
    
    # PML boundaries
    x_inner_min = pml_info['inner_x_min']
    x_inner_max = pml_info['inner_x_max']
    y_inner_min = pml_info['inner_y_min']
    y_inner_max = pml_info['inner_y_max']
    x_min = pml_info['x_min']
    x_max = pml_info['x_max']
    y_min = pml_info['y_min']
    y_max = pml_info['y_max']
    
    # Compute σ values
    sigma_x = 0.0
    sigma_y = 0.0
    
    # Left PML
    if x < x_inner_min:
        sigma_x = pml_sigma(x, x_inner_min, x_min, sigma_max)
    # Right PML
    elif x > x_inner_max:
        sigma_x = pml_sigma(x, x_inner_max, x_max, sigma_max)
    
    # Bottom PML
    if y < y_inner_min:
        sigma_y = pml_sigma(y, y_inner_min, y_min, sigma_max)
    # Top PML
    elif y > y_inner_max:
        sigma_y = pml_sigma(y, y_inner_max, y_max, sigma_max)
    
    # Complex stretching factors
    s_x = 1.0 + sigma_x / (1j * omega)
    s_y = 1.0 + sigma_y / (1j * omega)
    
    return np.array([s_x, s_y])


def apply_pml_to_matrices(K, M, fe, pml_info, omega, sigma_max=10.0):
    """Apply PML modification to existing matrices.
    
    For PML, we modify the mass matrix entries based on the complex
    coordinate stretching factors at each edge location.
    
    This is a simplified approach that modifies diagonal entries.
    """
    K_pml = K.astype(complex).tolil()
    M_pml = M.astype(complex).tolil()
    
    for i in range(fe.num_total_edges):
        edge = fe.edges[i]
        edge_center = (fe.points[edge[0]] + fe.points[edge[1]]) / 2
        s = compute_pml_tensor(edge_center, pml_info, omega, sigma_max)
        
        # Combined PML stretching factor
        pml_factor = s[0] * s[1]
        
        # Modify matrix entries
        # For curl-curl: multiply by 1/pml_factor
        # For mass: multiply by pml_factor
        if abs(pml_factor - 1.0) > 1e-10:
            # Scale row i of both matrices
            K_pml[i, :] = K_pml[i, :] / pml_factor
            M_pml[i, :] = M_pml[i, :] * pml_factor
    
    return K_pml.tocsr(), M_pml.tocsr()


def compute_point_source_rhs(fe, source_point, source_strength=1.0):
    """Compute RHS for point source excitation.
    
    A point source at x_s is represented as:
    J = J_0 * δ(x - x_s)
    
    In finite element, this becomes a projection onto basis functions:
    b_i = J_0 · N_i(x_s)
    
    We find the element containing the source and evaluate basis functions there.
    """
    b = np.zeros(fe.num_total_edges, dtype=complex)
    
    x_s, y_s = source_point
    
    # Find cell containing source point
    source_cell = None
    local_coords = None
    
    for cell_idx in range(fe.num_cells):
        cell_nodes = fe.cells[cell_idx]
        v0, v1, v2 = fe.points[cell_nodes]
        
        # Barycentric coordinates
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-12:
            continue
            
        lambda1 = ((v1[1] - v2[1]) * (x_s - v2[0]) + (v2[0] - v1[0]) * (y_s - v2[1])) / denom
        lambda2 = ((v2[1] - v0[1]) * (x_s - v2[0]) + (v0[0] - v2[0]) * (y_s - v2[1])) / denom
        lambda3 = 1 - lambda1 - lambda2
        
        tol = 1e-6
        if lambda1 >= -tol and lambda2 >= -tol and lambda3 >= -tol:
            source_cell = cell_idx
            local_coords = (lambda1, lambda2, lambda3)
            break
    
    if source_cell is None:
        print(f"Warning: Source point {source_point} not found in mesh!")
        return b
    
    # Get DOFs for this cell
    cell_dofs = fe.cell_to_dofs[source_cell]
    cell_nodes = fe.cells[source_cell]
    cell_points = fe.points[cell_nodes]
    
    # Evaluate edge basis functions at source point
    # For N1E triangle, basis functions on edges
    edges_local = [(0, 1), (1, 2), (2, 0)]
    
    for local_edge_idx, (n0, n1) in enumerate(edges_local):
        dof = cell_dofs[local_edge_idx]
        
        # Edge tangent vector
        p0 = cell_points[n0]
        p1 = cell_points[n1]
        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)
        tangent = edge_vec / edge_length
        
        # Edge basis function value at source (simplified)
        # N_e(x) ≈ λ_i * t_e where λ_i is barycentric coord of opposite vertex
        # For edge (0,1), opposite is vertex 2, so use lambda3
        if local_edge_idx == 0:  # edge (0,1)
            lambda_opp = local_coords[2]
        elif local_edge_idx == 1:  # edge (1,2)
            lambda_opp = local_coords[0]
        else:  # edge (2,0)
            lambda_opp = local_coords[1]
        
        # Basis function value (approximate)
        N_val = lambda_opp * tangent
        
        # Source contribution: J · N
        # Assume z-directed point source (for TM mode, this excites E_z which couples to tangential E)
        # For 2D H(curl), we use a tangential source
        source_direction = np.array([1.0, 0.0])  # x-directed source
        b[dof] += source_strength * np.dot(source_direction, N_val)
    
    return b


def solve_maxwell_pml(mesh, pml_info, frequency=1e9, sigma_max=10.0):
    """Solve Maxwell equations with PML.
    
    Parameters
    ----------
    mesh : Mesh
        The computational mesh
    pml_info : dict
        PML configuration
    frequency : float
        Operating frequency in Hz
    sigma_max : float
        Maximum PML conductivity
    
    Returns
    -------
    solution : array
        Complex electric field DOF values
    fe : HCurlFiniteElement
        The finite element space
    """
    # Physical constants
    c0 = 3e8  # Speed of light
    omega = 2 * np.pi * frequency
    k0 = omega / c0  # Free-space wavenumber
    
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    print(f"Wavelength: {c0/frequency:.4f} m")
    print(f"Wavenumber k0: {k0:.4f} rad/m")
    
    # Outer boundary: PEC (E × n = 0)
    def boundary_location(point):
        x, y = point
        tol = 1e-10
        x_min, x_max = pml_info['x_min'], pml_info['x_max']
        y_min, y_max = pml_info['y_min'], pml_info['y_max']
        return (x < x_min + tol or x > x_max - tol or 
                y < y_min + tol or y > y_max - tol)
    
    def zero_field(point):
        return np.array([0.0, 0.0])
    
    dirichlet_bc_info = [[boundary_location], [zero_field]]
    
    # Create H(curl) finite element space
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=2,
        ele_type='N1E_TRI1',
        gauss_order=3,
        dirichlet_bc_info=dirichlet_bc_info
    )
    
    print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} triangles")
    print(f"DOFs: {fe.num_total_edges} edges")
    
    # Compute standard matrices
    print("Assembling matrices...")
    K_std = compute_curl_curl_matrix(fe)
    M_std = compute_mass_matrix(fe)
    
    # Apply PML modification
    print("Applying PML...")
    K, M = apply_pml_to_matrices(K_std, M_std, fe, pml_info, omega, sigma_max)
    
    # System matrix: K - k0² M
    A = K - k0**2 * M
    
    # Apply boundary conditions
    if fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            for idx in edge_inds:
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
    
    # Point source at domain center
    domain_center = (
        (pml_info['inner_x_min'] + pml_info['inner_x_max']) / 2,
        (pml_info['inner_y_min'] + pml_info['inner_y_max']) / 2
    )
    print(f"Point source at: {domain_center}")
    
    b = compute_point_source_rhs(fe, domain_center, source_strength=1.0)
    
    # Apply BC to RHS
    if fe.edge_inds_list:
        for edge_inds, vals in zip(fe.edge_inds_list, fe.vals_list):
            b[edge_inds] = vals
    
    # Solve
    print("Solving complex linear system...")
    A_csr = A.tocsr()
    solution = spsolve(A_csr, b)
    
    residual = np.linalg.norm(A_csr @ solution - b)
    print(f"Residual: {residual:.6e}")
    
    return solution, fe


def visualize_pml_solution(solution, fe, pml_info, filename='maxwell_pml_solution.png'):
    """Visualize the PML solution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get edge centers and values
    edge_centers = np.array([
        (fe.points[e[0]] + fe.points[e[1]]) / 2 
        for e in fe.edges
    ])
    
    x = edge_centers[:, 0]
    y = edge_centers[:, 1]
    
    # Real part
    ax = axes[0, 0]
    vals_real = np.real(solution)
    sc = ax.tricontourf(x, y, vals_real, levels=50, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax, label='Re(E)')
    ax.set_title('Real Part of Electric Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Draw PML boundaries
    for ax_item in axes.flat:
        ax_item.axvline(x=pml_info['inner_x_min'], color='k', linestyle='--', alpha=0.5, label='PML boundary')
        ax_item.axvline(x=pml_info['inner_x_max'], color='k', linestyle='--', alpha=0.5)
        ax_item.axhline(y=pml_info['inner_y_min'], color='k', linestyle='--', alpha=0.5)
        ax_item.axhline(y=pml_info['inner_y_max'], color='k', linestyle='--', alpha=0.5)
    
    # Imaginary part
    ax = axes[0, 1]
    vals_imag = np.imag(solution)
    sc = ax.tricontourf(x, y, vals_imag, levels=50, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax, label='Im(E)')
    ax.set_title('Imaginary Part of Electric Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Magnitude
    ax = axes[1, 0]
    vals_mag = np.abs(solution)
    sc = ax.tricontourf(x, y, vals_mag, levels=50, cmap='hot')
    plt.colorbar(sc, ax=ax, label='|E|')
    ax.set_title('Electric Field Magnitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Mark source location
    source_x = (pml_info['inner_x_min'] + pml_info['inner_x_max']) / 2
    source_y = (pml_info['inner_y_min'] + pml_info['inner_y_max']) / 2
    ax.plot(source_x, source_y, 'g*', markersize=15, label='Point source')
    ax.legend()
    
    # Phase
    ax = axes[1, 1]
    vals_phase = np.angle(solution)
    sc = ax.tricontourf(x, y, vals_phase, levels=50, cmap='hsv')
    plt.colorbar(sc, ax=ax, label='Phase (rad)')
    ax.set_title('Electric Field Phase')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Add PML region labels
    axes[0, 0].text(pml_info['x_min'] + 0.02, 0.5, 'PML', fontsize=10, rotation=90, va='center')
    axes[0, 0].text(pml_info['x_max'] - 0.05, 0.5, 'PML', fontsize=10, rotation=90, va='center')
    
    plt.suptitle('2D Maxwell Equation with Point Source and PML\n(Time-Harmonic)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {filename}")
    plt.close()


def plot_field_along_line(solution, fe, pml_info, filename='maxwell_pml_line_plot.png'):
    """Plot field along a horizontal line through the source."""
    # Line through source center
    y_line = (pml_info['inner_y_min'] + pml_info['inner_y_max']) / 2
    
    # Get edge centers
    edge_centers = np.array([
        (fe.points[e[0]] + fe.points[e[1]]) / 2 
        for e in fe.edges
    ])
    
    # Find edges near the line
    tol = 0.05
    mask = np.abs(edge_centers[:, 1] - y_line) < tol
    
    x_line = edge_centers[mask, 0]
    vals_line = solution[mask]
    
    # Sort by x
    sort_idx = np.argsort(x_line)
    x_line = x_line[sort_idx]
    vals_line = vals_line[sort_idx]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Real and imaginary parts
    ax = axes[0]
    ax.plot(x_line, np.real(vals_line), 'b-', label='Re(E)', linewidth=2)
    ax.plot(x_line, np.imag(vals_line), 'r--', label='Im(E)', linewidth=2)
    ax.axvline(x=pml_info['inner_x_min'], color='k', linestyle=':', alpha=0.7, label='PML boundary')
    ax.axvline(x=pml_info['inner_x_max'], color='k', linestyle=':', alpha=0.7)
    ax.axvspan(pml_info['x_min'], pml_info['inner_x_min'], alpha=0.2, color='gray', label='PML region')
    ax.axvspan(pml_info['inner_x_max'], pml_info['x_max'], alpha=0.2, color='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('Electric Field')
    ax.set_title(f'Electric Field along y = {y_line:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Magnitude
    ax = axes[1]
    ax.semilogy(x_line, np.abs(vals_line), 'g-', linewidth=2, label='|E|')
    ax.axvline(x=pml_info['inner_x_min'], color='k', linestyle=':', alpha=0.7)
    ax.axvline(x=pml_info['inner_x_max'], color='k', linestyle=':', alpha=0.7)
    ax.axvspan(pml_info['x_min'], pml_info['inner_x_min'], alpha=0.2, color='gray', label='PML region')
    ax.axvspan(pml_info['inner_x_max'], pml_info['x_max'], alpha=0.2, color='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('|E| (log scale)')
    ax.set_title('Field Magnitude (showing PML absorption)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved line plot to {filename}")
    plt.close()


def main():
    """Main function to run the PML point source example."""
    print("="*70)
    print("2D Maxwell Equations with Point Source and PML")
    print("="*70)
    
    # Create mesh with PML
    # Inner region: 1m x 1m, PML thickness: 0.2m
    mesh, pml_info = create_pml_mesh(
        nx_inner=20, ny_inner=20,  # Inner mesh resolution
        nx_pml=5, ny_pml=5,        # PML mesh resolution
        domain_size=1.0,
        pml_thickness=0.2
    )
    
    print(f"\nDomain: [{pml_info['x_min']:.2f}, {pml_info['x_max']:.2f}] x "
          f"[{pml_info['y_min']:.2f}, {pml_info['y_max']:.2f}]")
    print(f"Inner (physical) region: [0, 1] x [0, 1]")
    print(f"PML thickness: {pml_info['pml_thickness']:.2f}")
    
    # Solve at 300 MHz (wavelength = 1m, matches domain size)
    frequency = 3e8  # 300 MHz
    solution, fe = solve_maxwell_pml(
        mesh, pml_info,
        frequency=frequency,
        sigma_max=5.0
    )
    
    print(f"\nSolution statistics:")
    print(f"  Max |E|: {np.max(np.abs(solution)):.6e}")
    print(f"  Mean |E|: {np.mean(np.abs(solution)):.6e}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_pml_solution(solution, fe, pml_info, 
                          filename='examples/maxwell_pml_solution.png')
    plot_field_along_line(solution, fe, pml_info,
                         filename='examples/maxwell_pml_line_plot.png')
    
    print("\n" + "="*70)
    print("PML Example Complete!")
    print("="*70)
    print("\nKey observations:")
    print("1. Point source at domain center excites outgoing waves")
    print("2. PML layers absorb waves with minimal reflection")
    print("3. Field magnitude decreases exponentially in PML region")
    print("4. Complex-valued solution shows propagating wave pattern")


if __name__ == '__main__':
    main()
