"""
2D Maxwell equations with dipole source and PML absorbing boundary.

This example demonstrates proper H(curl) Nedelec element implementation with:
1. Electric dipole source excitation
2. Perfectly Matched Layer (PML) via complex coordinate stretching
3. Time-harmonic Maxwell: curl(curl E) - k²E = -iωμ₀J

Author: JAX-FEM Team
"""

import os
# Configure JAX to use CPU with multi-threading
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16'

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from jax_fem.generate_mesh import Mesh
from jax_fem.hcurl_fe import (
    HCurlFiniteElement,
    compute_pml_matrices,
    compute_dipole_source_rhs,
    solve_complex_system,
)


def create_pml_mesh(nx_inner=40, ny_inner=40, nx_pml=8, ny_pml=8, 
                    domain_size=1.0, pml_thickness=0.25):
    """Create structured triangular mesh with PML regions."""
    # Total domain
    x_min = -pml_thickness
    x_max = domain_size + pml_thickness
    y_min = -pml_thickness
    y_max = domain_size + pml_thickness
    
    # Total elements
    total_nx = nx_inner + 2 * nx_pml
    total_ny = ny_inner + 2 * ny_pml
    
    # Create points
    x = np.linspace(x_min, x_max, total_nx + 1)
    y = np.linspace(y_min, y_max, total_ny + 1)
    
    points = []
    for j in range(total_ny + 1):
        for i in range(total_nx + 1):
            points.append([x[i], y[j]])
    points = np.array(points)
    
    # Create triangular cells
    cells = []
    for j in range(total_ny):
        for i in range(total_nx):
            n0 = j * (total_nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (total_nx + 1) + 1
            n3 = n0 + (total_nx + 1)
            cells.append([n0, n1, n2])
            cells.append([n0, n2, n3])
    cells = np.array(cells)
    
    mesh = Mesh(points, cells)
    
    pml_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'inner_x_min': 0.0, 'inner_x_max': domain_size,
        'inner_y_min': 0.0, 'inner_y_max': domain_size,
        'pml_thickness': pml_thickness,
    }
    
    return mesh, pml_info


def create_pml_sigma_func(pml_info, sigma_max=2.0, order=3):
    """Create PML conductivity function.
    
    Uses polynomial grading: σ(d) = σ_max * (d/L)^order
    where d is distance into PML and L is PML thickness.
    """
    x_inner_min = pml_info['inner_x_min']
    x_inner_max = pml_info['inner_x_max']
    y_inner_min = pml_info['inner_y_min']
    y_inner_max = pml_info['inner_y_max']
    L = pml_info['pml_thickness']
    
    def pml_sigma(x, y):
        sigma_x = 0.0
        sigma_y = 0.0
        
        # Left PML
        if x < x_inner_min:
            d = (x_inner_min - x) / L
            sigma_x = sigma_max * (d ** order)
        # Right PML
        elif x > x_inner_max:
            d = (x - x_inner_max) / L
            sigma_x = sigma_max * (d ** order)
        
        # Bottom PML
        if y < y_inner_min:
            d = (y_inner_min - y) / L
            sigma_y = sigma_max * (d ** order)
        # Top PML
        elif y > y_inner_max:
            d = (y - y_inner_max) / L
            sigma_y = sigma_max * (d ** order)
        
        return sigma_x, sigma_y
    
    return pml_sigma


def solve_maxwell_pml(mesh, pml_info, frequency=1e9, sigma_max=2.0):
    """Solve time-harmonic Maxwell with PML."""
    
    # Physical constants
    c0 = 3e8
    omega = 2 * np.pi * frequency
    k0 = omega / c0
    wavelength = c0 / frequency
    
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    print(f"Wavelength: {wavelength:.4f} m")
    print(f"Wavenumber k0: {k0:.2f} rad/m")
    print(f"Wavelengths in domain: {1.0/wavelength:.1f}")
    
    # Boundary condition: PEC on outer boundary
    def outer_boundary(point):
        x, y = point
        tol = 1e-10
        return (abs(x - pml_info['x_min']) < tol or 
                abs(x - pml_info['x_max']) < tol or
                abs(y - pml_info['y_min']) < tol or 
                abs(y - pml_info['y_max']) < tol)
    
    def zero_tangent(point):
        return 0.0
    
    dirichlet_bc_info = [[outer_boundary], [zero_tangent]]
    
    # Create H(curl) finite element
    fe = HCurlFiniteElement(
        mesh=mesh,
        dim=2,
        ele_type='N1E_TRI1',
        gauss_order=3,
        dirichlet_bc_info=dirichlet_bc_info
    )
    
    print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} triangles")
    print(f"DOFs: {fe.num_total_dofs} edges")
    
    # PML function
    pml_sigma = create_pml_sigma_func(pml_info, sigma_max)
    
    # Assemble PML matrices
    print("Assembling PML matrices...")
    K_pml, M_pml = compute_pml_matrices(fe, pml_sigma, omega)
    
    # System: K - k0² M
    A = K_pml - k0**2 * M_pml
    
    # Dipole source at center, x-polarized
    source_point = [0.5, 0.5]
    polarization = [1.0, 0.0]  # x-polarized
    
    # Gaussian width ~ 2-3 element sizes
    h = 1.0 / 40  # Approximate element size
    sigma_source = 3 * h
    
    print(f"Dipole source at: {source_point}, polarization: {polarization}")
    b = compute_dipole_source_rhs(fe, source_point, polarization, 
                                   amplitude=1.0, sigma=sigma_source)
    
    # Apply Dirichlet BC
    if hasattr(fe, 'edge_inds_list') and fe.edge_inds_list:
        for edge_inds in fe.edge_inds_list:
            for idx in edge_inds:
                A[idx, :] = 0
                A[:, idx] = 0
                A[idx, idx] = 1.0
                b[idx] = 0
    
    # Solve complex linear system
    print("Solving complex linear system...")
    import time as time_module
    
    A_csr = A.tocsr()
    start_solve = time_module.time()
    
    # Use HYPRE AMS parallel solver (optimized for H(curl) problems)
    # Options: 'hypre_ams' (parallel, recommended), 'pardiso', 'direct', 'petsc_mumps'
    solution = solve_complex_system(A_csr, b, method='hypre_ams', fe=fe)
    
    solve_time = time_module.time() - start_solve
    print(f"Total solve time: {solve_time:.2f}s")
    
    residual = np.linalg.norm(A_csr @ solution - b)
    print(f"Residual: {residual:.6e}")
    
    return solution, fe, k0


def visualize_solution(solution, fe, pml_info, k0, filename='examples/maxwell_pml_solution.png'):
    """Visualize solution with circular wave pattern."""
    from scipy.interpolate import griddata
    
    # Get edge centers
    edge_centers = np.array([
        (fe.points[e[0]] + fe.points[e[1]]) / 2 
        for e in fe.edges
    ])
    
    x = edge_centers[:, 0]
    y = edge_centers[:, 1]
    
    # Fine grid for interpolation
    xi = np.linspace(pml_info['x_min'], pml_info['x_max'], 300)
    yi = np.linspace(pml_info['y_min'], pml_info['y_max'], 300)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    vals_real = np.real(solution)
    vals_mag = np.abs(solution)
    
    Zi_real = griddata((x, y), vals_real, (Xi, Yi), method='cubic')
    Zi_mag = griddata((x, y), vals_mag, (Xi, Yi), method='cubic')
    
    source_x, source_y = 0.5, 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Real part
    ax = axes[0, 0]
    vmax = np.nanpercentile(np.abs(Zi_real), 95)
    sc = ax.pcolormesh(Xi, Yi, Zi_real, cmap='RdBu_r', shading='auto',
                       vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='Re(E)')
    ax.set_title('Real Part - Wave Pattern (H(curl) Nedelec)', fontsize=12)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.plot(source_x, source_y, 'k*', markersize=15, label='Dipole')
    ax.legend()
    
    # Draw PML boundaries
    for ax_item in axes.flat:
        ax_item.axvline(x=pml_info['inner_x_min'], color='w', linestyle='--', linewidth=2)
        ax_item.axvline(x=pml_info['inner_x_max'], color='w', linestyle='--', linewidth=2)
        ax_item.axhline(y=pml_info['inner_y_min'], color='w', linestyle='--', linewidth=2)
        ax_item.axhline(y=pml_info['inner_y_max'], color='w', linestyle='--', linewidth=2)
        ax_item.set_xlim(pml_info['x_min'], pml_info['x_max'])
        ax_item.set_ylim(pml_info['y_min'], pml_info['y_max'])
    
    # Imaginary part
    ax = axes[0, 1]
    vals_imag = np.imag(solution)
    Zi_imag = griddata((x, y), vals_imag, (Xi, Yi), method='cubic')
    vmax = np.nanpercentile(np.abs(Zi_imag), 95)
    sc = ax.pcolormesh(Xi, Yi, Zi_imag, cmap='RdBu_r', shading='auto',
                       vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='Im(E)')
    ax.set_title('Imaginary Part', fontsize=12)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.plot(source_x, source_y, 'k*', markersize=15)
    
    # Magnitude
    ax = axes[1, 0]
    sc = ax.pcolormesh(Xi, Yi, Zi_mag, cmap='hot', shading='auto')
    plt.colorbar(sc, ax=ax, label='|E|')
    ax.set_title('Field Magnitude', fontsize=12)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.plot(source_x, source_y, 'g*', markersize=15, label='Dipole')
    ax.legend()
    
    # Shade PML regions
    for ax_item in [axes[1, 0], axes[1, 1]]:
        ax_item.axvspan(pml_info['x_min'], pml_info['inner_x_min'], alpha=0.2, color='blue')
        ax_item.axvspan(pml_info['inner_x_max'], pml_info['x_max'], alpha=0.2, color='blue')
        ax_item.axhspan(pml_info['y_min'], pml_info['inner_y_min'], alpha=0.2, color='blue')
        ax_item.axhspan(pml_info['inner_y_max'], pml_info['y_max'], alpha=0.2, color='blue')
    
    # Log magnitude
    ax = axes[1, 1]
    Zi_log = np.log10(np.maximum(Zi_mag, 1e-12))
    sc = ax.pcolormesh(Xi, Yi, Zi_log, cmap='viridis', shading='auto')
    plt.colorbar(sc, ax=ax, label='log₁₀|E|')
    ax.set_title('Log Magnitude - PML Absorption', fontsize=12)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.plot(source_x, source_y, 'r*', markersize=15)
    
    wavelength = 2 * np.pi / k0
    plt.suptitle(f'2D Maxwell: Dipole Source + PML (Nedelec H(curl))\n'
                 f'λ = {wavelength:.3f} m, k = {k0:.1f} rad/m', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_field_decay(solution, fe, pml_info, filename='examples/maxwell_pml_decay.png'):
    """Plot field decay along a line through source."""
    source_x, source_y = 0.5, 0.5
    
    # Get edge centers
    edge_centers = np.array([
        (fe.points[e[0]] + fe.points[e[1]]) / 2 
        for e in fe.edges
    ])
    
    # Find edges along y = 0.5
    tol = 0.03
    mask = np.abs(edge_centers[:, 1] - source_y) < tol
    x_line = edge_centers[mask, 0]
    vals = np.abs(solution[mask])
    
    idx = np.argsort(x_line)
    x_line = x_line[idx]
    vals = vals[idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(x_line, vals, 'b-', linewidth=2, label='|E(x, 0.5)|')
    
    # Mark PML regions
    ax.axvline(x=pml_info['inner_x_min'], color='g', linestyle='--', label='PML boundary')
    ax.axvline(x=pml_info['inner_x_max'], color='g', linestyle='--')
    ax.axvspan(pml_info['x_min'], pml_info['inner_x_min'], alpha=0.2, color='green', label='PML')
    ax.axvspan(pml_info['inner_x_max'], pml_info['x_max'], alpha=0.2, color='green')
    
    # Mark source
    ax.axvline(x=source_x, color='r', linestyle=':', linewidth=2, label='Source')
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('|E|', fontsize=12)
    ax.set_title('Field Decay with PML Absorption', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pml_info['x_min'], pml_info['x_max'])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    print("="*70)
    print("2D Maxwell: Dipole Source with PML (H(curl) Nedelec Elements)")
    print("="*70)
    
    # Use moderate frequency for clear wave patterns
    # λ = c/f = 3e8 / 1e9 = 0.3 m -> ~3 wavelengths in 1m domain
    frequency = 1e9  # 1 GHz
    
    mesh, pml_info = create_pml_mesh(
        nx_inner=50, ny_inner=50,  # Finer mesh
        nx_pml=10, ny_pml=10,
        domain_size=1.0,
        pml_thickness=0.25
    )
    
    print(f"\nDomain: [{pml_info['x_min']:.2f}, {pml_info['x_max']:.2f}] x "
          f"[{pml_info['y_min']:.2f}, {pml_info['y_max']:.2f}]")
    print(f"Physical region: [0, 1] x [0, 1]")
    print(f"PML thickness: {pml_info['pml_thickness']:.2f}")
    
    solution, fe, k0 = solve_maxwell_pml(
        mesh, pml_info,
        frequency=frequency,
        sigma_max=3.0
    )
    
    print(f"\nSolution statistics:")
    print(f"  Max |E|: {np.max(np.abs(solution)):.6e}")
    print(f"  Mean |E|: {np.mean(np.abs(solution)):.6e}")
    
    print("\nGenerating visualizations...")
    visualize_solution(solution, fe, pml_info, k0)
    plot_field_decay(solution, fe, pml_info)
    
    print("\n" + "="*70)
    print("PML Example Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
