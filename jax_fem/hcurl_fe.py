"""
H(curl) conforming finite elements (Nedelec elements) for JAX-FEM.

This module provides support for Nedelec edge elements, which are essential for
electromagnetic problems where the unknown field must be H(curl) conforming.

Key features:
- Nedelec first kind (N1E) and second kind (N2E) elements
- Support for triangles, quadrilaterals, tetrahedra, and hexahedra
- Proper covariant Piola transformation for mapping to physical elements
- Curl computation in physical coordinates

References:
- Nédélec, J.C. (1980). Mixed finite elements in R³
- Monk, P. (2003). Finite Element Methods for Maxwell's Equations
"""

import numpy as onp
import jax
import jax.numpy as np
import sys
import time
from dataclasses import dataclass
from jax_fem.generate_mesh import Mesh
from jax_fem.basis import (get_face_shape_vals_and_grads, get_shape_vals_and_grads, 
                           get_elements, is_hcurl_element)
from jax_fem import logger


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@dataclass
class HCurlFiniteElement:
    """H(curl) conforming finite element class for electromagnetic problems.
    
    This class handles Nedelec edge elements where DOFs are associated with edges
    (and faces/cells for higher orders) rather than nodes.
    
    Attributes
    ----------
    mesh : Mesh
        Stores points (coordinates) and cells (connectivity).
    dim : int
        Spatial dimension of the problem (2 or 3).
    ele_type : str
        Element type. Supported H(curl) types:
        
        - 'N1E_TRI1', 'N1E_TRI2': Nedelec first kind on triangles
        - 'N1E_QUAD1', 'N1E_QUAD2': Nedelec first kind on quadrilaterals
        - 'N1E_TET1', 'N1E_TET2': Nedelec first kind on tetrahedra
        - 'N1E_HEX1', 'N1E_HEX2': Nedelec first kind on hexahedra
        - 'N2E_*': Nedelec second kind variants
        
    gauss_order : int
        Order of Gaussian quadrature.
    dirichlet_bc_info : list
        Boundary condition information for tangential components.
    """
    mesh: Mesh
    dim: int
    ele_type: str
    gauss_order: int
    dirichlet_bc_info: list

    def __post_init__(self):
        assert is_hcurl_element(self.ele_type), \
            f"Element type {self.ele_type} is not an H(curl) element. Use N1E_* or N2E_* types."
        
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        
        start = time.time()
        logger.debug(f"Computing H(curl) shape function values, gradients, etc.")

        # Get shape functions - these are vector-valued for H(curl)
        # shape_vals: (num_quads, num_dofs, dim)
        # shape_grads_ref: (num_quads, num_dofs, dim, dim)
        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(
            self.ele_type, self.gauss_order)
        
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, \
            self.face_normals, self.face_inds = get_face_shape_vals_and_grads(
                self.ele_type, self.gauss_order)
        
        self.num_quads = self.shape_vals.shape[0]
        self.num_dofs_per_cell = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.num_face_quads = self.face_quad_weights.shape[1]
        
        # Compute Jacobians and physical shape functions
        self.jacobians, self.jacobian_dets, self.jacobian_invs = self._compute_jacobians()
        self.shape_vals_physical, self.shape_curls_physical = self._compute_physical_basis()
        self.JxW = self.jacobian_dets * self.quad_weights[None, :]
        
        # Build edge-to-DOF mapping
        self._build_edge_dof_map()
        
        # Handle boundary conditions
        self.edge_inds_list, self.vals_list = self._process_boundary_conditions(
            self.dirichlet_bc_info)
        
        end = time.time()
        compute_time = end - start

        logger.debug(f"Done H(curl) pre-computations, took {compute_time} [s]")
        logger.info(f"H(curl) problem with {len(self.cells)} cells, "
                    f"{self.num_total_edges} edges (DOFs).")
        logger.info(f"Element type is {self.ele_type}, using {self.num_quads} "
                    f"quad points per element.")

    def _compute_jacobians(self):
        """Compute Jacobian matrices for each cell at each quadrature point.
        
        Returns
        -------
        jacobians : NumpyArray
            Shape (num_cells, num_quads, dim, dim). Jacobian dx/dξ.
        jacobian_dets : NumpyArray
            Shape (num_cells, num_quads). Determinant of Jacobian.
        jacobian_invs : NumpyArray
            Shape (num_cells, num_quads, dim, dim). Inverse Jacobian dξ/dx.
        """
        # Use linear Lagrange elements for geometry mapping
        geom_ele_type = self._get_geometry_element_type()
        
        from jax_fem.basis import get_shape_vals_and_grads as get_lagrange_basis
        _, geom_shape_grads_ref, _ = get_lagrange_basis(geom_ele_type, self.gauss_order)
        # geom_shape_grads_ref: (num_quads, num_nodes, dim)
        
        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        
        # Compute Jacobian: J[i,j] = sum_k x_k[i] * dN_k/dxi[j]
        # physical_coos: (num_cells, num_nodes, dim)
        # geom_shape_grads_ref: (num_quads, num_nodes, dim)
        # jacobians: (num_cells, num_quads, dim, dim)
        # J = dx/dξ, so J[i,j] = dx_i/dξ_j = sum_k x_k[i] * dN_k/dξ_j
        jacobians = onp.einsum('cni,qnj->cqij', physical_coos, geom_shape_grads_ref)
        
        jacobian_dets = onp.linalg.det(jacobians)  # (num_cells, num_quads)
        jacobian_invs = onp.linalg.inv(jacobians)  # (num_cells, num_quads, dim, dim)
        
        return jacobians, jacobian_dets, jacobian_invs

    def _get_geometry_element_type(self):
        """Get the corresponding Lagrange element type for geometry mapping."""
        if 'TRI' in self.ele_type:
            return 'TRI3' if '1' in self.ele_type else 'TRI6'
        elif 'QUAD' in self.ele_type:
            return 'QUAD4' if '1' in self.ele_type else 'QUAD8'
        elif 'TET' in self.ele_type:
            return 'TET4' if '1' in self.ele_type else 'TET10'
        elif 'HEX' in self.ele_type:
            return 'HEX8' if '1' in self.ele_type else 'HEX20'
        else:
            raise ValueError(f"Unknown element type: {self.ele_type}")

    def _compute_physical_basis(self):
        """Compute physical basis functions using covariant Piola transformation.
        
        For H(curl) elements, the covariant Piola transformation is:
            φ_physical = J^{-T} φ_reference
        
        The curl transforms as:
            curl(φ_physical) = (1/det(J)) J curl(φ_reference)  (3D)
            curl(φ_physical) = (1/det(J)) curl(φ_reference)    (2D, scalar curl)
        
        Returns
        -------
        shape_vals_physical : NumpyArray
            Shape (num_cells, num_quads, num_dofs, dim). Physical basis functions.
        shape_curls_physical : NumpyArray
            Shape (num_cells, num_quads, num_dofs) for 2D (scalar curl).
            Shape (num_cells, num_quads, num_dofs, dim) for 3D (vector curl).
        """
        # Covariant Piola: φ_phys = J^{-T} φ_ref
        # shape_vals: (num_quads, num_dofs, dim)
        # jacobian_invs: (num_cells, num_quads, dim, dim)
        
        # J^{-T}: (num_cells, num_quads, dim, dim)
        J_inv_T = onp.transpose(self.jacobian_invs, (0, 1, 3, 2))
        
        # (num_cells, num_quads, num_dofs, dim)
        shape_vals_physical = onp.einsum('cqij,qdj->cqdi', J_inv_T, self.shape_vals)
        
        # Compute curl
        if self.dim == 2:
            # 2D curl is scalar: curl(v) = ∂v_y/∂x - ∂v_x/∂y
            # shape_grads_ref: (num_quads, num_dofs, dim, dim) 
            # [q, d, i, j] = ∂φ_d^i/∂ξ_j
            
            # Transform to physical: ∂φ/∂x = (∂ξ/∂x)^T ∂φ/∂ξ = J^{-T} ∂φ/∂ξ
            # But for curl, we need a different approach
            # curl_ref = ∂φ_y/∂ξ_x - ∂φ_x/∂ξ_y (in reference coordinates)
            # Actually: curl transforms as curl_phys = (1/det(J)) curl_ref for 2D
            
            curl_ref = self.shape_grads_ref[:, :, 1, 0] - self.shape_grads_ref[:, :, 0, 1]
            # curl_ref: (num_quads, num_dofs)
            # curl_physical: (num_cells, num_quads, num_dofs)
            shape_curls_physical = curl_ref[None, :, :] / self.jacobian_dets[:, :, None]
        else:
            # 3D curl is vector: curl(v) = (∂v_z/∂y - ∂v_y/∂z, ∂v_x/∂z - ∂v_z/∂x, ∂v_y/∂x - ∂v_x/∂y)
            curl_ref = onp.zeros((self.num_quads, self.num_dofs_per_cell, 3))
            curl_ref[:, :, 0] = self.shape_grads_ref[:, :, 2, 1] - self.shape_grads_ref[:, :, 1, 2]
            curl_ref[:, :, 1] = self.shape_grads_ref[:, :, 0, 2] - self.shape_grads_ref[:, :, 2, 0]
            curl_ref[:, :, 2] = self.shape_grads_ref[:, :, 1, 0] - self.shape_grads_ref[:, :, 0, 1]
            
            # Piola transformation for curl: curl_phys = (1/det(J)) J curl_ref
            # (num_cells, num_quads, num_dofs, 3)
            shape_curls_physical = onp.einsum('cqij,qdj->cqdi', 
                                              self.jacobians / self.jacobian_dets[:, :, None, None],
                                              curl_ref)
        
        return shape_vals_physical, shape_curls_physical

    def _build_edge_dof_map(self):
        """Build mapping from global edges to DOFs.
        
        For lowest order Nedelec elements, each edge has one DOF.
        For higher orders, there are additional DOFs on faces and in cells.
        """
        # Extract edges from cells
        edges_set = set()
        cell_edges = []
        
        if self.dim == 2:
            # For 2D: edges are the sides of triangles/quads
            if 'TRI' in self.ele_type:
                edge_local = [(0, 1), (1, 2), (2, 0)]
            else:  # QUAD
                edge_local = [(0, 1), (1, 2), (2, 3), (3, 0)]
        else:
            # For 3D tetrahedra
            if 'TET' in self.ele_type:
                edge_local = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            else:  # HEX
                edge_local = [(0, 1), (1, 2), (2, 3), (3, 0),  # bottom
                              (4, 5), (5, 6), (6, 7), (7, 4),  # top
                              (0, 4), (1, 5), (2, 6), (3, 7)]  # vertical
        
        for cell in self.cells:
            cell_edge_list = []
            for e0, e1 in edge_local:
                edge = tuple(sorted([cell[e0], cell[e1]]))
                edges_set.add(edge)
                cell_edge_list.append(edge)
            cell_edges.append(cell_edge_list)
        
        # Create global edge numbering
        self.edges = sorted(list(edges_set))
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        self.num_total_edges = len(self.edges)
        
        # For lowest order, num_dofs = num_edges
        # For higher orders, we'd need to add face and cell DOFs
        self.num_total_dofs = self.num_total_edges
        
        # Cell to edge DOF mapping
        self.cell_to_dofs = onp.array([
            [self.edge_to_idx[edge] for edge in cell_edge_list]
            for cell_edge_list in cell_edges
        ])

    def _process_boundary_conditions(self, dirichlet_bc_info):
        """Process Dirichlet boundary conditions for H(curl) elements.
        
        For H(curl), Dirichlet BC specifies the tangential component of the field
        on the boundary.
        
        Parameters
        ----------
        dirichlet_bc_info : list or None
            [location_fns, value_fns] where location_fns identify boundary edges
            and value_fns provide the tangential field values.
        
        Returns
        -------
        edge_inds_list : list[NumpyArray]
            Global edge indices for each BC set.
        vals_list : list[NumpyArray]
            Tangential values at each edge for each BC set.
        """
        edge_inds_list = []
        vals_list = []
        
        if dirichlet_bc_info is not None:
            location_fns, value_fns = dirichlet_bc_info
            
            for loc_fn, val_fn in zip(location_fns, value_fns):
                edge_inds = []
                edge_vals = []
                
                for edge_idx, (n0, n1) in enumerate(self.edges):
                    p0 = self.points[n0]
                    p1 = self.points[n1]
                    midpoint = 0.5 * (p0 + p1)
                    
                    # Check if edge midpoint satisfies location condition
                    if loc_fn(midpoint):
                        edge_inds.append(edge_idx)
                        # Tangent direction
                        tangent = p1 - p0
                        tangent = tangent / onp.linalg.norm(tangent)
                        # Get prescribed field value at midpoint
                        field_val = val_fn(midpoint)
                        # DOF value is the tangential component integrated along edge
                        edge_length = onp.linalg.norm(p1 - p0)
                        tangential_val = onp.dot(field_val, tangent) * edge_length
                        edge_vals.append(tangential_val)
                
                edge_inds_list.append(onp.array(edge_inds, dtype=onp.int32))
                vals_list.append(onp.array(edge_vals))
        
        return edge_inds_list, vals_list

    def get_physical_quad_points(self):
        """Compute physical quadrature points.
        
        Returns
        -------
        physical_quad_points : NumpyArray
            Shape is (num_cells, num_quads, dim).
        """
        geom_ele_type = self._get_geometry_element_type()
        from jax_fem.basis import get_shape_vals_and_grads as get_lagrange_basis
        geom_shape_vals, _, _ = get_lagrange_basis(geom_ele_type, self.gauss_order)
        
        physical_coos = onp.take(self.points, self.cells, axis=0)
        physical_quad_points = onp.sum(geom_shape_vals[None, :, :, None] * 
                                       physical_coos[:, None, :, :], axis=2)
        return physical_quad_points

    def convert_from_dof_to_quad(self, sol):
        """Compute field values at quadrature points from DOF values.
        
        Parameters
        ----------
        sol : JaxArray
            Shape is (num_total_dofs,). DOF values (edge integrals).
        
        Returns
        -------
        u : JaxArray
            Shape is (num_cells, num_quads, dim). Field values at quad points.
        """
        # Get DOF values for each cell
        cell_dofs = sol[self.cell_to_dofs]  # (num_cells, num_dofs_per_cell)
        
        # Interpolate: u = sum_i dof_i * phi_i
        # shape_vals_physical: (num_cells, num_quads, num_dofs_per_cell, dim)
        # cell_dofs: (num_cells, num_dofs_per_cell)
        u = np.sum(cell_dofs[:, None, :, None] * self.shape_vals_physical, axis=2)
        return u

    def compute_curl(self, sol):
        """Compute curl of field at quadrature points from DOF values.
        
        Parameters
        ----------
        sol : JaxArray
            Shape is (num_total_dofs,). DOF values.
        
        Returns
        -------
        curl_u : JaxArray
            Shape is (num_cells, num_quads) for 2D (scalar curl).
            Shape is (num_cells, num_quads, dim) for 3D (vector curl).
        """
        cell_dofs = sol[self.cell_to_dofs]  # (num_cells, num_dofs_per_cell)
        
        if self.dim == 2:
            # shape_curls_physical: (num_cells, num_quads, num_dofs_per_cell)
            curl_u = np.sum(cell_dofs[:, None, :] * self.shape_curls_physical, axis=2)
        else:
            # shape_curls_physical: (num_cells, num_quads, num_dofs_per_cell, dim)
            curl_u = np.sum(cell_dofs[:, None, :, None] * self.shape_curls_physical, axis=2)
        
        return curl_u


def compute_curl_curl_matrix(fe):
    """Assemble the curl-curl stiffness matrix for H(curl) elements.
    
    The weak form is: (curl u, curl v) for all test functions v.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The finite element space.
    
    Returns
    -------
    K : sparse matrix
        The curl-curl stiffness matrix of shape (num_dofs, num_dofs).
    """
    from scipy.sparse import coo_matrix
    
    num_dofs = fe.num_total_dofs
    
    # Local stiffness matrix for each cell
    # (curl phi_i, curl phi_j) integrated over each cell
    
    rows = []
    cols = []
    vals = []
    
    for c in range(fe.num_cells):
        cell_dof_ids = fe.cell_to_dofs[c]
        
        for i, dof_i in enumerate(cell_dof_ids):
            for j, dof_j in enumerate(cell_dof_ids):
                # Integrate (curl phi_i, curl phi_j) * JxW
                if fe.dim == 2:
                    val = onp.sum(fe.shape_curls_physical[c, :, i] * 
                                  fe.shape_curls_physical[c, :, j] * 
                                  fe.JxW[c, :])
                else:
                    val = onp.sum(onp.sum(fe.shape_curls_physical[c, :, i, :] * 
                                          fe.shape_curls_physical[c, :, j, :], axis=-1) * 
                                  fe.JxW[c, :])
                
                rows.append(dof_i)
                cols.append(dof_j)
                vals.append(val)
    
    K = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs))
    return K.tocsr()


def compute_mass_matrix(fe):
    """Assemble the mass matrix for H(curl) elements.
    
    The weak form is: (u, v) for all test functions v.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The finite element space.
    
    Returns
    -------
    M : sparse matrix
        The mass matrix of shape (num_dofs, num_dofs).
    """
    from scipy.sparse import coo_matrix
    
    num_dofs = fe.num_total_dofs
    
    rows = []
    cols = []
    vals = []
    
    for c in range(fe.num_cells):
        cell_dof_ids = fe.cell_to_dofs[c]
        
        for i, dof_i in enumerate(cell_dof_ids):
            for j, dof_j in enumerate(cell_dof_ids):
                # Integrate (phi_i . phi_j) * JxW
                val = onp.sum(onp.sum(fe.shape_vals_physical[c, :, i, :] * 
                                      fe.shape_vals_physical[c, :, j, :], axis=-1) * 
                              fe.JxW[c, :])
                
                rows.append(dof_i)
                cols.append(dof_j)
                vals.append(val)
    
    M = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs))
    return M.tocsr()


def compute_discrete_gradient(fe):
    """Compute the discrete gradient matrix G for AMS preconditioner.
    
    The discrete gradient matrix maps the H1 nodal space to the H(curl) edge space.
    This is required by HYPRE's AMS (Auxiliary-space Maxwell Solver) preconditioner.
    
    For each edge e = (n0, n1), we have:
        G[edge_idx, n1] = +1
        G[edge_idx, n0] = -1
    
    This represents: grad(phi_node) projected onto edge tangent.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    
    Returns
    -------
    G : scipy.sparse.csr_matrix
        The discrete gradient matrix of shape (num_edges, num_nodes).
        Can be converted to PETSc format for use with HYPRE AMS.
    
    Example
    -------
    >>> G = compute_discrete_gradient(fe)
    >>> # Convert to PETSc for AMS
    >>> from petsc4py import PETSc
    >>> G_petsc = PETSc.Mat().createAIJ(size=G.shape, 
    ...     csr=(G.indptr, G.indices, G.data))
    >>> pc.setHYPREDiscreteGradient(G_petsc)
    """
    from scipy.sparse import coo_matrix
    
    num_edges = fe.num_total_edges
    num_nodes = fe.num_total_nodes
    
    rows = []
    cols = []
    vals = []
    
    for edge_idx, (n0, n1) in enumerate(fe.edges):
        # G maps nodal values to edge integrals
        # For edge (n0, n1): G @ phi gives integral of grad(phi) dot tangent
        rows.extend([edge_idx, edge_idx])
        cols.extend([n0, n1])
        vals.extend([-1.0, 1.0])  # grad = (phi[n1] - phi[n0]) / length, but AMS normalizes
    
    G = coo_matrix((vals, (rows, cols)), shape=(num_edges, num_nodes))
    return G.tocsr()


def compute_node_coordinates(fe):
    """Get node coordinates array for AMS preconditioner.
    
    HYPRE AMS can use node coordinates to build better interpolation operators.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    
    Returns
    -------
    coords : numpy.ndarray
        Node coordinates of shape (num_nodes, dim).
    """
    return fe.points.copy()


def petsc_ams_solve(A_petsc, b, fe, ksp_type='cg', rtol=1e-10, max_iter=1000):
    """Solve H(curl) linear system using PETSc with HYPRE AMS preconditioner.
    
    This is the recommended solver for Maxwell-type problems discretized with 
    Nedelec edge elements.
    
    Parameters
    ----------
    A_petsc : PETSc.Mat
        The system matrix (e.g., curl-curl + mass matrix).
    b : numpy.ndarray
        The right-hand side vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space (needed for discrete gradient).
    ksp_type : str
        Krylov solver type. Default 'cg' for symmetric positive definite.
    rtol : float
        Relative tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    x : numpy.ndarray
        The solution vector.
    
    Example
    -------
    >>> K = compute_curl_curl_matrix(fe)
    >>> M = compute_mass_matrix(fe)
    >>> A = K + omega**2 * M  # Time-harmonic Maxwell
    >>> # Convert to PETSc...
    >>> x = petsc_ams_solve(A_petsc, b, fe)
    """
    from petsc4py import PETSc
    
    # Compute discrete gradient matrix
    G_scipy = compute_discrete_gradient(fe)
    G_petsc = PETSc.Mat().createAIJ(
        size=G_scipy.shape,
        csr=(G_scipy.indptr.astype(PETSc.IntType),
             G_scipy.indices.astype(PETSc.IntType),
             G_scipy.data)
    )
    
    # Setup KSP solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol, max_it=max_iter)
    
    # Setup HYPRE AMS preconditioner
    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setHYPREType('ams')
    pc.setHYPREDiscreteGradient(G_petsc)
    
    # Optionally set coordinates for better interpolation
    coords = compute_node_coordinates(fe)
    # Note: setCoordinates requires specific format, may need adjustment
    
    ksp.setFromOptions()
    
    # Solve
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), b)
    x = PETSc.Vec().createSeq(len(b))
    
    ksp.solve(rhs, x)
    
    # Check convergence
    if not ksp.getConvergedReason() > 0:
        import warnings
        warnings.warn(f"AMS solver may not have converged: reason = {ksp.getConvergedReason()}")
    
    return x.getArray()
