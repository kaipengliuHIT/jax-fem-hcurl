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

    def build_discrete_gradient_matrix(self):
        """Build discrete gradient matrix G for AMS preconditioner.
        
        The discrete gradient matrix G maps from nodal (H1) DOFs to edge (H(curl)) DOFs.
        For each edge connecting nodes i and j, G has entries:
            G[edge_idx, i] = -1
            G[edge_idx, j] = +1
        
        This represents the discrete gradient operator: grad(phi) on edges.
        
        Returns
        -------
        G : scipy.sparse.csr_matrix
            Discrete gradient matrix of shape (num_edges, num_nodes).
        """
        from scipy.sparse import lil_matrix
        
        num_edges = self.num_total_edges
        num_nodes = self.mesh.points.shape[0]
        
        G = lil_matrix((num_edges, num_nodes))
        
        # For each edge, set gradient entries
        for edge_idx, (node_i, node_j) in enumerate(self.edges):
            G[edge_idx, node_i] = -1.0
            G[edge_idx, node_j] = +1.0
        
        return G.tocsr()

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


def compute_curl_curl_matrix(fe, use_gpu=True):
    """Assemble the curl-curl stiffness matrix for H(curl) elements using JAX GPU.
    
    The weak form is: (curl u, curl v) for all test functions v.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The finite element space.
    use_gpu : bool
        Whether to use JAX GPU acceleration (default True).
    
    Returns
    -------
    K : sparse matrix
        The curl-curl stiffness matrix of shape (num_dofs, num_dofs).
    """
    from scipy.sparse import coo_matrix
    import time as time_module
    
    start_time = time_module.time()
    num_dofs = fe.num_total_dofs
    num_cells = fe.num_cells
    
    if use_gpu and jax.devices()[0].platform == 'gpu':
        logger.info("Using JAX GPU for curl-curl matrix assembly")
        
        shape_curls = np.array(fe.shape_curls_physical)
        JxW = np.array(fe.JxW)
        
        @jax.jit
        def compute_local_stiffness_2d(shape_curls, JxW):
            curl_outer = shape_curls[:, :, :, None] * shape_curls[:, :, None, :]
            return np.einsum('cqij,cq->cij', curl_outer, JxW)
        
        @jax.jit
        def compute_local_stiffness_3d(shape_curls, JxW):
            curl_dot = np.einsum('cqid,cqjd->cqij', shape_curls, shape_curls)
            return np.einsum('cqij,cq->cij', curl_dot, JxW)
        
        if fe.dim == 2:
            K_local = onp.array(compute_local_stiffness_2d(shape_curls, JxW))
        else:
            K_local = onp.array(compute_local_stiffness_3d(shape_curls, JxW))
        
        gpu_time = time_module.time() - start_time
        logger.info(f"GPU curl-curl computation: {gpu_time:.4f} s")
    else:
        logger.info("Using NumPy CPU for curl-curl matrix assembly")
        shape_curls = fe.shape_curls_physical
        JxW = fe.JxW
        
        if fe.dim == 2:
            curl_outer = shape_curls[:, :, :, None] * shape_curls[:, :, None, :]
            K_local = onp.einsum('cqij,cq->cij', curl_outer, JxW)
        else:
            curl_dot = onp.einsum('cqid,cqjd->cqij', shape_curls, shape_curls)
            K_local = onp.einsum('cqij,cq->cij', curl_dot, JxW)
    
    # Vectorized global assembly using precomputed DOF indices
    cell_dofs = onp.array(fe.cell_to_dofs)  # (num_cells, dofs_per_cell)
    num_local_dofs = cell_dofs.shape[1]
    
    # Create row/col indices for all cells at once
    rows = onp.repeat(cell_dofs[:, :, None], num_local_dofs, axis=2).ravel()
    cols = onp.repeat(cell_dofs[:, None, :], num_local_dofs, axis=1).ravel()
    vals = K_local.ravel()
    
    K = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs))
    
    total_time = time_module.time() - start_time
    logger.info(f"Curl-curl matrix assembly total: {total_time:.4f} s")
    
    return K.tocsr()


def compute_mass_matrix(fe, use_gpu=True):
    """Assemble the mass matrix for H(curl) elements using JAX GPU.
    
    The weak form is: (u, v) for all test functions v.
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The finite element space.
    use_gpu : bool
        Whether to use JAX GPU acceleration (default True).
    
    Returns
    -------
    M : sparse matrix
        The mass matrix of shape (num_dofs, num_dofs).
    """
    from scipy.sparse import coo_matrix
    import time as time_module
    
    start_time = time_module.time()
    num_dofs = fe.num_total_dofs
    num_cells = fe.num_cells
    
    if use_gpu and jax.devices()[0].platform == 'gpu':
        logger.info("Using JAX GPU for mass matrix assembly")
        
        shape_vals = np.array(fe.shape_vals_physical)
        JxW = np.array(fe.JxW)
        
        @jax.jit
        def compute_local_mass(shape_vals, JxW):
            # shape_vals: (cells, quads, dofs, dim)
            # Dot product over dim, then sum over quads
            vals_dot = np.einsum('cqid,cqjd->cqij', shape_vals, shape_vals)
            return np.einsum('cqij,cq->cij', vals_dot, JxW)
        
        M_local = onp.array(compute_local_mass(shape_vals, JxW))
        
        gpu_time = time_module.time() - start_time
        logger.info(f"GPU mass matrix computation: {gpu_time:.4f} s")
    else:
        logger.info("Using NumPy CPU for mass matrix assembly")
        shape_vals = fe.shape_vals_physical
        JxW = fe.JxW
        
        vals_dot = onp.einsum('cqid,cqjd->cqij', shape_vals, shape_vals)
        M_local = onp.einsum('cqij,cq->cij', vals_dot, JxW)
    
    # Vectorized global assembly
    cell_dofs = onp.array(fe.cell_to_dofs)
    num_local_dofs = cell_dofs.shape[1]
    
    rows = onp.repeat(cell_dofs[:, :, None], num_local_dofs, axis=2).ravel()
    cols = onp.repeat(cell_dofs[:, None, :], num_local_dofs, axis=1).ravel()
    vals = M_local.ravel()
    
    M = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs))
    
    total_time = time_module.time() - start_time
    logger.info(f"Mass matrix assembly total: {total_time:.4f} s")
    
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


def compute_pml_matrices(fe, pml_func, omega, use_gpu=True):
    """Assemble curl-curl and mass matrices with PML using JAX GPU acceleration.
    
    For time-harmonic Maxwell equations with PML, the weak form becomes:
    
        ∫ (Λ^{-1} curl E) · (curl v) dΩ - k² ∫ (Λ E) · v dΩ = -iωμ₀ ∫ J · v dΩ
    
    where Λ is the PML tensor based on complex coordinate stretching:
        s_i = 1 + σ_i / (iω)
        
    For 2D TM mode:
        Λ = diag(s_y/s_x, s_x/s_y)  for mass term
        Λ^{-1} = diag(s_x/s_y, s_y/s_x)  for curl-curl term
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    pml_func : callable
        Function that returns (sigma_x, sigma_y) at a given point (x, y).
        Should return (0, 0) in the physical domain and positive values in PML.
    omega : float
        Angular frequency (2 * pi * f).
    use_gpu : bool
        Whether to use JAX GPU acceleration (default True).
    
    Returns
    -------
    K_pml : sparse matrix (complex)
        The PML-modified curl-curl stiffness matrix.
    M_pml : sparse matrix (complex)
        The PML-modified mass matrix.
    """
    from scipy.sparse import coo_matrix
    import time as time_module
    
    num_dofs = fe.num_total_dofs
    num_cells = fe.num_cells
    num_dofs_per_cell = fe.num_dofs_per_cell
    num_quads = fe.num_quads
    
    start_time = time_module.time()
    
    # Compute cell centroids
    cell_coords = fe.points[fe.cells]  # (num_cells, nodes_per_cell, dim)
    centroids = onp.mean(cell_coords, axis=1)  # (num_cells, dim)
    
    # Compute PML sigma values for all cells
    sigma_x = onp.zeros(num_cells)
    sigma_y = onp.zeros(num_cells)
    for c in range(num_cells):
        sigma_x[c], sigma_y[c] = pml_func(centroids[c, 0], centroids[c, 1])
    
    # Complex stretching factors (num_cells,)
    s_x = 1.0 + sigma_x / (1j * omega)
    s_y = 1.0 + sigma_y / (1j * omega)
    
    # PML tensors for all cells
    lambda_curl = s_x * s_y  # (num_cells,) - scalar for 2D curl
    lambda_mass_0 = s_y / s_x  # (num_cells,) - x-component
    lambda_mass_1 = s_x / s_y  # (num_cells,) - y-component
    
    if use_gpu and jax.devices()[0].platform == 'gpu':
        logger.info("Using JAX GPU for PML matrix assembly")
        
        # Convert to JAX arrays on GPU
        shape_curls = np.array(fe.shape_curls_physical)  # (cells, quads, dofs)
        shape_vals = np.array(fe.shape_vals_physical)    # (cells, quads, dofs, dim)
        JxW = np.array(fe.JxW)  # (cells, quads)
        lambda_curl_j = np.array(lambda_curl)
        lambda_mass_0_j = np.array(lambda_mass_0)
        lambda_mass_1_j = np.array(lambda_mass_1)
        
        # Vectorized element matrix computation on GPU
        @jax.jit
        def compute_element_matrices(shape_curls, shape_vals, JxW, 
                                      lambda_curl, lambda_mass_0, lambda_mass_1):
            # shape_curls: (cells, quads, dofs)
            # shape_vals: (cells, quads, dofs, dim)
            # JxW: (cells, quads)
            
            # Curl-curl: K_ij = lambda_curl * sum_q(curl_i * curl_j * JxW)
            # Outer product over dofs, sum over quads
            curl_outer = shape_curls[:, :, :, None] * shape_curls[:, :, None, :]  # (cells, quads, dofs, dofs)
            K_local = np.einsum('cqij,cq,c->cij', curl_outer, JxW, lambda_curl)
            
            # Mass: M_ij = sum_d(lambda_mass_d * sum_q(N_i_d * N_j_d * JxW))
            # x-component
            vals_x_outer = shape_vals[:, :, :, 0, None] * shape_vals[:, :, None, :, 0]  # (cells, quads, dofs, dofs)
            M_x = np.einsum('cqij,cq,c->cij', vals_x_outer, JxW, lambda_mass_0)
            
            # y-component
            vals_y_outer = shape_vals[:, :, :, 1, None] * shape_vals[:, :, None, :, 1]
            M_y = np.einsum('cqij,cq,c->cij', vals_y_outer, JxW, lambda_mass_1)
            
            M_local = M_x + M_y
            
            return K_local, M_local
        
        # Run on GPU
        K_local, M_local = compute_element_matrices(
            shape_curls, shape_vals, JxW,
            lambda_curl_j, lambda_mass_0_j, lambda_mass_1_j
        )
        
        # Block until computation is done
        K_local = onp.array(K_local)
        M_local = onp.array(M_local)
        
        gpu_time = time_module.time() - start_time
        logger.info(f"GPU element matrix computation: {gpu_time:.4f} s")
        
    else:
        logger.info("Using NumPy CPU for PML matrix assembly")
        
        # Vectorized CPU computation using einsum
        shape_curls = fe.shape_curls_physical  # (cells, quads, dofs)
        shape_vals = fe.shape_vals_physical    # (cells, quads, dofs, dim)
        JxW = fe.JxW
        
        # Curl-curl matrices
        curl_outer = shape_curls[:, :, :, None] * shape_curls[:, :, None, :]
        K_local = onp.einsum('cqij,cq,c->cij', curl_outer, JxW, lambda_curl)
        
        # Mass matrices
        vals_x_outer = shape_vals[:, :, :, 0, None] * shape_vals[:, :, None, :, 0]
        vals_y_outer = shape_vals[:, :, :, 1, None] * shape_vals[:, :, None, :, 1]
        M_x = onp.einsum('cqij,cq,c->cij', vals_x_outer, JxW, lambda_mass_0)
        M_y = onp.einsum('cqij,cq,c->cij', vals_y_outer, JxW, lambda_mass_1)
        M_local = M_x + M_y
    
    # Assemble global matrices
    assemble_start = time_module.time()
    
    # Vectorized global assembly
    cell_dofs = onp.array(fe.cell_to_dofs)
    num_local_dofs = cell_dofs.shape[1]
    
    rows = onp.repeat(cell_dofs[:, :, None], num_local_dofs, axis=2).ravel()
    cols = onp.repeat(cell_dofs[:, None, :], num_local_dofs, axis=1).ravel()
    
    K_pml = coo_matrix((K_local.ravel(), (rows, cols)), shape=(num_dofs, num_dofs), dtype=complex)
    M_pml = coo_matrix((M_local.ravel(), (rows, cols)), shape=(num_dofs, num_dofs), dtype=complex)
    
    total_time = time_module.time() - start_time
    logger.info(f"PML matrix assembly total: {total_time:.4f} s")
    
    return K_pml.tocsr(), M_pml.tocsr()


def build_block_complex_system(A_complex, b_complex):
    """Convert complex system to real 2x2 block system for use with real preconditioners.
    
    For a complex system (A_r + i*A_i)(x_r + i*x_i) = b_r + i*b_i,
    the equivalent real block system is:
    
        [A_r  -A_i] [x_r]   [b_r]
        [A_i   A_r] [x_i] = [b_i]
    
    This allows using HYPRE AMS (real-only) as a block diagonal preconditioner.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    
    Returns
    -------
    A_block : sparse matrix (real)
        The 2x2 block real system matrix.
    b_block : numpy.ndarray (real)
        The block RHS vector [b_r; b_i].
    n : int
        Size of original system (for extracting solution).
    """
    from scipy.sparse import bmat, csr_matrix
    
    n = A_complex.shape[0]
    
    # Extract real and imaginary parts
    A_r = A_complex.real.tocsr()
    A_i = A_complex.imag.tocsr()
    
    # Build block matrix: [A_r, -A_i; A_i, A_r]
    A_block = bmat([[A_r, -A_i], 
                    [A_i, A_r]], format='csr')
    
    # Build block RHS
    b_r = b_complex.real
    b_i = b_complex.imag
    b_block = onp.concatenate([b_r, b_i])
    
    return A_block, b_block, n


def extract_complex_solution(x_block, n):
    """Extract complex solution from block system solution.
    
    Parameters
    ----------
    x_block : numpy.ndarray
        Solution of block system [x_r; x_i].
    n : int
        Size of original complex system.
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution x_r + i*x_i.
    """
    x_r = x_block[:n]
    x_i = x_block[n:]
    return x_r + 1j * x_i


def build_block_preconditioner_matrix(A_complex):
    """Build preconditioner matrix for block complex system.
    
    For the preconditioner, we use the absolute value form:
        P = [|A|, 0; 0, |A|]
    where |A| = sqrt(A_r² + A_i²) element-wise.
    
    This is a common approach used in MFEM for complex PML problems.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    
    Returns
    -------
    P_diag : sparse matrix (real)
        The diagonal block for preconditioning (|A|).
    """
    A_r = A_complex.real
    A_i = A_complex.imag
    
    # Element-wise absolute value
    A_abs_data = onp.sqrt(A_r.data**2 + A_i.data**2)
    
    from scipy.sparse import csr_matrix
    P_diag = csr_matrix((A_abs_data, A_r.indices, A_r.indptr), shape=A_r.shape)
    
    return P_diag


def _solve_hypre_ams(A_complex, b_complex, fe, tol=1e-8, maxiter=500):
    """Solve H(curl) system using HYPRE AMS preconditioner with GMRES.
    
    AMS (Auxiliary-space Maxwell Solver) is specifically designed for H(curl) problems.
    It uses MPI parallelization for multi-core acceleration.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix (curl-curl - k²·mass).
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space (provides discrete gradient matrix).
    tol : float
        Relative tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    
    Returns
    -------
    solution : numpy.ndarray (complex)
        The solution vector.
    """
    from petsc4py import PETSc
    import time as time_module
    
    start_time = time_module.time()
    
    # Build real 2x2 block system for HYPRE (real matrices only)
    logger.info("Building 2x2 block real system for HYPRE AMS...")
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    n_block = A_block.shape[0]
    
    # Convert to PETSc format
    # Note: For true MPI parallelism, matrix assembly must be distributed
    # Current implementation: sequential assembly, but HYPRE can still use threads
    A_csr = A_block.tocsr()
    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(A_csr.indptr.astype(PETSc.IntType),
             A_csr.indices.astype(PETSc.IntType),
             A_csr.data)
    )
    
    # Create vectors
    rhs = PETSc.Vec().createSeq(n_block)
    rhs.setValues(range(n_block), b_block)
    x = PETSc.Vec().createSeq(n_block)
    
    # Build discrete gradient matrix G for AMS
    logger.info("Building discrete gradient matrix for AMS...")
    G = fe.build_discrete_gradient_matrix()
    
    # Create block G matrix for real system: [G, 0; 0, G]
    from scipy.sparse import bmat
    G_block = bmat([[G, None], [None, G]], format='csr')
    
    G_petsc = PETSc.Mat().createAIJ(
        size=G_block.shape,
        csr=(G_block.indptr.astype(PETSc.IntType),
             G_block.indices.astype(PETSc.IntType),
             G_block.data)
    )
    
    # Get vertex coordinates for AMS
    coords = fe.mesh.points  # (num_nodes, dim)
    n_nodes = coords.shape[0]
    dim = coords.shape[1]
    
    # Create coordinate vectors for block system
    # For 2D: x, y coordinates; For 3D: x, y, z
    coord_vecs = []
    for d in range(dim):
        # Duplicate for real/imag blocks
        coord_data = onp.concatenate([coords[:, d], coords[:, d]])
        vec = PETSc.Vec().createSeq(2 * n_nodes)
        vec.setValues(range(2 * n_nodes), coord_data)
        coord_vecs.append(vec)
    
    # Setup KSP with HYPRE AMS preconditioner
    logger.info(f"Setting up HYPRE GMRES + AMS preconditioner ({n_block} DOFs)...")
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType('gmres')
    ksp.setTolerances(rtol=tol, max_it=maxiter)
    
    # Set GMRES restart and orthogonalization
    ksp.setGMRESRestart(100)  # Restart every 100 iterations
    
    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setHYPREType('ams')
    
    # Set AMS discrete gradient
    pc.setHYPREDiscreteGradient(G_petsc)
    
    # Set vertex coordinates
    if dim == 2:
        pc.setHYPRESetEdgeConstantVectors(coord_vecs[0], coord_vecs[1])
    else:
        pc.setHYPRESetEdgeConstantVectors(coord_vecs[0], coord_vecs[1], coord_vecs[2])
    
    # Configure HYPRE AMS options
    import os
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    if num_threads > 1:
        logger.info(f"Setting OMP_NUM_THREADS={num_threads} for HYPRE...")
    
    # Set HYPRE AMS options via PETSc
    # Use BoomerAMG for the internal solves
    PETSc.Options().setValue('-pc_hypre_ams_cycle_type', 1)  # 1-cycle
    PETSc.Options().setValue('-pc_hypre_ams_print_level', 1)  # Print info
    
    pc.setFromOptions()
    ksp.setFromOptions()
    
    logger.info(f"Solving with HYPRE AMS ({n_block} DOFs)...")
    ksp.solve(rhs, x)
    
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    solve_time = time_module.time() - start_time
    
    if reason > 0:
        logger.info(f"HYPRE AMS converged in {its} iterations, {solve_time:.2f}s")
    else:
        logger.warning(f"HYPRE AMS did not converge: reason={reason}, iterations={its}")
    
    # Extract complex solution
    x_block = x.getArray()
    solution = extract_complex_solution(x_block, n_orig)
    
    # Check residual
    residual = onp.linalg.norm(A_complex @ solution - b_complex)
    logger.info(f"HYPRE AMS residual: {residual:.2e}")
    
    return solution


def _solve_petsc_mumps(A_complex, b_complex):
    """Solve complex system using PETSc MUMPS parallel direct solver.
    
    MUMPS is a parallel sparse direct solver that uses MPI for parallelization.
    It works well for H(curl) problems with complex matrices.
    """
    from petsc4py import PETSc
    import time as time_module
    
    start_time = time_module.time()
    n = A_complex.shape[0]
    
    logger.info(f"Building complex system for PETSc MUMPS ({n} DOFs)...")
    
    # PETSc can handle complex matrices directly
    A_csr = A_complex.tocsr()
    
    # Create PETSc matrix (complex)
    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(A_csr.indptr.astype(PETSc.IntType),
             A_csr.indices.astype(PETSc.IntType),
             A_csr.data)
    )
    
    # Create vectors
    rhs = PETSc.Vec().createSeq(n)
    rhs.setValues(range(n), b_complex)
    x = PETSc.Vec().createSeq(n)
    
    # Setup KSP with MUMPS direct solver
    logger.info("Setting up PETSc MUMPS parallel direct solver...")
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType('preonly')  # Direct solver, no Krylov iteration
    
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')  # Use MUMPS
    
    # Configure MUMPS for parallel execution
    # ICNTL(14): percentage increase in working space (default 20)
    # ICNTL(7): ordering (7=automatic)
    pc.setFromOptions()
    
    logger.info(f"Solving with MUMPS ({n} complex DOFs)...")
    ksp.solve(rhs, x)
    
    reason = ksp.getConvergedReason()
    solve_time = time_module.time() - start_time
    
    if reason > 0:
        logger.info(f"MUMPS solver completed in {solve_time:.2f}s")
    else:
        logger.warning(f"MUMPS solver issue: reason={reason}")
    
    solution = x.getArray()
    
    # Check residual
    residual = onp.linalg.norm(A_csr @ solution - b_complex)
    logger.info(f"MUMPS residual: {residual:.2e}")
    
    return solution


def solve_petsc_gpu(A_complex, b_complex, fe, ksp_type='gmres', pc_type='jacobi', tol=1e-6, maxiter=2000):
    """Solve complex H(curl) using PETSc GPU solver with cuSPARSE.
    
    Uses PETSc's native CUDA support for GPU-accelerated sparse solve.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    ksp_type : str
        Krylov solver type.
    pc_type : str
        Preconditioner type.
    tol : float
        Relative tolerance.
    maxiter : int
        Maximum iterations.
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution.
    """
    from petsc4py import PETSc
    from scipy.sparse import bmat
    import time as time_module
    import os
    
    start_time = time_module.time()
    
    # Configure PETSc for GPU
    os.environ['PETSC_OPTIONS'] = '-use_gpu_aware_mpi 0 -vec_type cuda -mat_type aijcusparse'
    
    logger.info("Building 2x2 block real system for PETSc GPU solver...")
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    
    # Convert to PETSc format
    logger.info("Converting to PETSc CUDA format...")
    A_csr = A_block.tocsr()
    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(A_csr.indptr.astype(PETSc.IntType),
             A_csr.indices.astype(PETSc.IntType),
             A_csr.data)
    )
    
    # Create CUDA vectors
    rhs = PETSc.Vec().createSeq(len(b_block))
    rhs.setValues(range(len(b_block)), b_block)
    x = PETSc.Vec().createSeq(len(b_block))
    
    # Setup KSP solver
    logger.info(f"Setting up PETSc GPU {ksp_type.upper()} solver with {pc_type.upper()} preconditioner...")
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=tol, max_it=maxiter)
    
    pc = ksp.getPC()
    pc.setType(pc_type)
    
    ksp.setFromOptions()
    
    logger.info(f"Solving on GPU ({len(b_block)} DOFs)...")
    ksp.solve(rhs, x)
    
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    
    solve_time = time_module.time() - start_time
    if reason > 0:
        logger.info(f"PETSc GPU solver converged in {its} iterations, {solve_time:.2f}s")
    else:
        logger.warning(f"PETSc GPU solver did not converge: reason={reason}, iterations={its}")
    
    # Extract solution
    x_block = x.getArray()
    solution = extract_complex_solution(x_block, n_orig)
    
    return solution


def solve_complex_system(A_complex, b_complex, method='direct', use_gpu=True, fe=None):
    """Solve complex linear system using various methods.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    method : str
        'direct' for scipy spsolve, 
        'gpu_gmres' for JAX GPU GMRES with Jacobi precond,
        'gpu_bicgstab' for JAX GPU BiCGSTAB,
        'jax_gmres_ams' for JAX GMRES + HYPRE AMS preconditioner (recommended for H(curl)),
        'gmres' for CPU GMRES.
    use_gpu : bool
        Whether to use GPU if available (default True).
    fe : HCurlFiniteElement, optional
        Required for 'jax_gmres_ams' method.
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution.
    """
    from scipy.sparse.linalg import spsolve, gmres
    import time as time_module
    
    start_time = time_module.time()
    
    # Check if GPU available
    gpu_available = use_gpu and jax.devices()[0].platform == 'gpu'
    
    if method == 'petsc_gpu':
        if fe is None:
            raise ValueError("fe (HCurlFiniteElement) required for petsc_gpu method")
        logger.info("Solving with PETSc GPU (cuSPARSE)...")
        solution = solve_petsc_gpu(A_complex, b_complex, fe)
        solve_time = time_module.time() - start_time
        logger.info(f"Total solve time: {solve_time:.2f}s")
        return solution
    
    elif method == 'jax_gpu':
        if fe is None:
            raise ValueError("fe (HCurlFiniteElement) required for jax_gpu method")
        logger.info("Solving with pure JAX GPU BiCGSTAB (block real system)...")
        solution = solve_hcurl_complex_jax_gpu(A_complex, b_complex, fe)
        solve_time = time_module.time() - start_time
        logger.info(f"Total solve time: {solve_time:.2f}s")
        return solution
    
    elif method == 'jax_gpu_ams':
        if fe is None:
            raise ValueError("fe (HCurlFiniteElement) required for jax_gpu_ams method")
        logger.info("Solving with JAX GPU GMRES + CPU AMS preconditioner...")
        solution = solve_hcurl_complex_jax_gpu_with_ams(A_complex, b_complex, fe)
        solve_time = time_module.time() - start_time
        logger.info(f"Total solve time: {solve_time:.2f}s")
        return solution
    
    elif method == 'gpu_bicgstab' and gpu_available:
        logger.info("Solving with JAX GPU BiCGSTAB (sparse BCOO)...")
        solution = _solve_gpu_sparse_bicgstab(A_complex, b_complex)
        
    elif method == 'gpu_gmres' and gpu_available:
        logger.info("Solving with JAX GPU GMRES (sparse BCOO)...")
        solution = _solve_gpu_gmres_sparse(A_complex, b_complex)
        
    elif method == 'pardiso':
        logger.info("Solving complex system with MKL PARDISO (multi-threaded)...")
        try:
            import os
            os.environ.setdefault('MKL_NUM_THREADS', '16')
            from pypardiso import spsolve as pardiso_solve
            # Build real block system for PARDISO (handles complex)
            A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
            x_block = pardiso_solve(A_block.tocsr(), b_block)
            solution = extract_complex_solution(x_block, n_orig)
        except ImportError:
            logger.warning("pypardiso not installed, falling back to spsolve")
            solution = spsolve(A_complex.tocsr(), b_complex)
    
    elif method == 'hypre_ams':
        if fe is None:
            raise ValueError("fe (HCurlFiniteElement) required for hypre_ams method")
        logger.info("Solving with HYPRE AMS (parallel H(curl) preconditioner)...")
        solution = _solve_hypre_ams(A_complex, b_complex, fe)
        solve_time = time_module.time() - start_time
        logger.info(f"Total solve time: {solve_time:.2f}s")
        return solution
    
    elif method == 'petsc_mumps':
        logger.info("Solving with PETSc MUMPS (parallel direct solver)...")
        solution = _solve_petsc_mumps(A_complex, b_complex)
    
    elif method == 'direct':
        logger.info("Solving complex system with CPU direct solver (spsolve)...")
        solution = spsolve(A_complex.tocsr(), b_complex)
        
    elif method == 'gmres':
        logger.info("Solving complex system with CPU GMRES...")
        from scipy.sparse.linalg import spilu, LinearOperator
        
        A_csr = A_complex.tocsr()
        n = A_csr.shape[0]
        
        try:
            ilu = spilu(A_csr.tocsc())
            M = LinearOperator((n, n), matvec=ilu.solve, dtype=complex)
            solution, info = gmres(A_csr, b_complex, M=M, rtol=1e-8, maxiter=500)
            if info != 0:
                logger.warning(f"GMRES did not converge: info={info}")
        except Exception as e:
            logger.warning(f"GMRES failed ({e}), using direct solver")
            solution = spsolve(A_csr, b_complex)
    else:
        # Fallback to direct
        logger.info("Falling back to CPU direct solver...")
        solution = spsolve(A_complex.tocsr(), b_complex)
    
    solve_time = time_module.time() - start_time
    logger.info(f"Solve completed in {solve_time:.2f}s")
    
    return solution


def _solve_gpu_sparse_bicgstab(A_complex, b_complex, tol=1e-8, maxiter=1000):
    """Solve complex system using JAX GPU BiCGSTAB with SPARSE matrix.
    
    This is the correct way to use JAX GPU - using BCOO sparse format
    like the original JAX-FEM project does in solver.py.
    """
    from jax.experimental.sparse import BCOO
    import scipy.sparse
    
    n = A_complex.shape[0]
    logger.info(f"Converting to JAX BCOO sparse format for GPU solve ({n}x{n})...")
    
    # Convert complex sparse matrix to real block system for JAX
    # (JAX bicgstab works better with real systems)
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    
    # Convert scipy sparse to JAX BCOO sparse format (key step!)
    A_csr = A_block.tocsr()
    A_bcoo = BCOO.from_scipy_sparse(A_csr).sort_indices()
    
    # Transfer RHS to GPU
    b_gpu = np.array(b_block)
    
    # Jacobi preconditioner (diagonal scaling)
    jacobi = np.array(A_csr.diagonal())
    jacobi = np.where(np.abs(jacobi) > 1e-14, jacobi, 1.0)  # Avoid division by zero
    
    def precond(x):
        return x / jacobi
    
    logger.info(f"Running JAX GPU BiCGSTAB on {2*n_orig}x{2*n_orig} sparse block system...")
    
    # JAX sparse BiCGSTAB - this runs on GPU!
    x_gpu, info = jax.scipy.sparse.linalg.bicgstab(
        A_bcoo, 
        b_gpu, 
        x0=None,
        M=precond,
        tol=tol, 
        atol=tol,
        maxiter=maxiter
    )
    x_gpu.block_until_ready()
    
    # Check residual
    residual = np.linalg.norm(A_bcoo @ x_gpu - b_gpu)
    logger.info(f"GPU BiCGSTAB residual: {residual:.2e}")
    
    if residual > 0.1:
        logger.warning(f"GPU solver may not have converged, residual={residual:.2e}")
    
    # Extract complex solution
    x_block = onp.array(x_gpu)
    solution = extract_complex_solution(x_block, n_orig)
    
    return solution


def _solve_gpu_gmres_sparse(A_complex, b_complex, tol=1e-8, maxiter=500):
    """Solve complex system using JAX GPU GMRES with SPARSE matrix.
    
    Uses real block formulation for complex system.
    """
    from jax.experimental.sparse import BCOO
    
    n = A_complex.shape[0]
    logger.info(f"Converting to JAX BCOO sparse format for GPU GMRES ({n}x{n})...")
    
    # Convert to real block system
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    
    # Convert to JAX sparse
    A_csr = A_block.tocsr()
    A_bcoo = BCOO.from_scipy_sparse(A_csr).sort_indices()
    
    b_gpu = np.array(b_block)
    
    # Jacobi preconditioner
    jacobi = np.array(A_csr.diagonal())
    jacobi = np.where(np.abs(jacobi) > 1e-14, jacobi, 1.0)
    
    def precond(x):
        return x / jacobi
    
    logger.info(f"Running JAX GPU GMRES on {2*n_orig}x{2*n_orig} sparse block system...")
    
    # JAX sparse GMRES with preconditioner
    x_gpu, info = jax.scipy.sparse.linalg.gmres(
        A_bcoo, 
        b_gpu,
        M=precond,
        tol=tol, 
        maxiter=maxiter
    )
    x_gpu.block_until_ready()
    
    residual = np.linalg.norm(A_bcoo @ x_gpu - b_gpu)
    logger.info(f"GPU GMRES residual: {residual:.2e}")
    
    x_block = onp.array(x_gpu)
    solution = extract_complex_solution(x_block, n_orig)
    
    return solution


def build_ams_preconditioner_matrix(A_complex, fe, num_sweeps=1):
    """Build AMS preconditioner matrix on CPU using HYPRE.
    
    This computes M_ams ≈ A^{-1} using HYPRE AMS, which can then be
    transferred to GPU for use in JAX iterative solvers.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    num_sweeps : int
        Number of AMS smoothing sweeps.
    
    Returns
    -------
    M_ams : sparse matrix (real)
        The AMS preconditioner matrix for the real block system.
    """
    from scipy.sparse import bmat, identity
    from petsc4py import PETSc
    import time as time_module
    
    start_time = time_module.time()
    n = A_complex.shape[0]
    
    logger.info(f"Building AMS preconditioner matrix on CPU ({n} DOFs)...")
    
    # Build real 2x2 block system
    A_block, _, n_orig = build_block_complex_system(A_complex, onp.zeros(n, dtype=complex))
    
    # Build preconditioner matrix (absolute value form)
    P_diag = build_block_preconditioner_matrix(A_complex)
    P_block = bmat([[P_diag, None], [None, P_diag]], format='csr')
    
    # Setup HYPRE AMS
    P_petsc = PETSc.Mat().createAIJ(
        size=P_block.shape,
        csr=(P_block.indptr.astype(PETSc.IntType),
             P_block.indices.astype(PETSc.IntType),
             P_block.data)
    )
    
    G_scipy = compute_discrete_gradient(fe)
    G_block = bmat([[G_scipy, None], [None, G_scipy]], format='csr')
    G_petsc = PETSc.Mat().createAIJ(
        size=G_block.shape,
        csr=(G_block.indptr.astype(PETSc.IntType),
             G_block.indices.astype(PETSc.IntType),
             G_block.data)
    )
    
    ksp_pc = PETSc.KSP().create()
    ksp_pc.setOperators(P_petsc)
    ksp_pc.setType('preonly')
    
    pc = ksp_pc.getPC()
    pc.setType('hypre')
    pc.setHYPREType('ams')
    pc.setHYPREDiscreteGradient(G_petsc)
    
    coords = compute_node_coordinates(fe)
    coords_block = onp.vstack([coords, coords])
    pc.setCoordinates(coords_block)
    
    pc.setUp()
    
    # Apply AMS to identity matrix to get approximate inverse
    n_block = 2 * n_orig
    logger.info(f"Computing AMS approximate inverse ({n_block}x{n_block})...")
    
    # For efficiency, we'll just use AMS as a function, not build full matrix
    # Return the PC object for later use
    build_time = time_module.time() - start_time
    logger.info(f"AMS preconditioner setup completed in {build_time:.2f}s")
    
    return pc, n_block


def solve_hcurl_complex_jax_gpu_with_ams(A_complex, b_complex, fe, tol=1e-8, maxiter=1000):
    """Solve complex H(curl) using JAX GPU GMRES with CPU-computed AMS preconditioner.
    
    Strategy:
    1. Build AMS preconditioner on CPU (one-time cost)
    2. Use JAX GPU GMRES for iterations
    3. Apply AMS preconditioner via callback (some GPU-CPU transfer per iteration)
    
    This is a compromise: most computation on GPU, but preconditioner on CPU.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix.
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    tol : float
        Relative tolerance.
    maxiter : int
        Maximum iterations.
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution.
    """
    from jax.experimental.sparse import BCOO
    import time as time_module
    
    start_time = time_module.time()
    n = A_complex.shape[0]
    
    # Build AMS preconditioner on CPU
    pc_ams, n_block = build_ams_preconditioner_matrix(A_complex, fe)
    
    # Build real block system
    logger.info(f"Building 2x2 block real system...")
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    
    # Convert to JAX BCOO for GPU
    logger.info("Converting to JAX BCOO sparse format for GPU...")
    A_csr = A_block.tocsr()
    A_bcoo = BCOO.from_scipy_sparse(A_csr).sort_indices()
    b_gpu = np.array(b_block)
    
    # Create preconditioner function using jax.pure_callback
    # Note: This will cause some GPU-CPU transfer per iteration
    from petsc4py import PETSc
    
    def ams_precond_host(r_np):
        """Apply AMS preconditioner on CPU"""
        r_petsc = PETSc.Vec().createSeq(n_block)
        r_petsc.setValues(range(n_block), r_np)
        z_petsc = PETSc.Vec().createSeq(n_block)
        pc_ams.apply(r_petsc, z_petsc)
        return z_petsc.getArray().astype(onp.float64)
    
    def ams_precond(r):
        """JAX-compatible AMS preconditioner"""
        return jax.pure_callback(
            ams_precond_host,
            jax.ShapeDtypeStruct((n_block,), r.dtype),
            r
        )
    
    logger.info(f"Running JAX GPU GMRES with AMS preconditioner ({n_block} DOFs)...")
    logger.info("Note: Preconditioner application involves GPU-CPU transfer per iteration")
    
    # JAX GMRES on GPU with CPU preconditioner
    x_gpu, info = jax.scipy.sparse.linalg.gmres(
        A_bcoo,
        b_gpu,
        M=ams_precond,
        tol=tol,
        maxiter=maxiter
    )
    x_gpu.block_until_ready()
    
    residual = np.linalg.norm(A_bcoo @ x_gpu - b_gpu)
    solve_time = time_module.time() - start_time
    logger.info(f"JAX GPU GMRES + AMS: residual={residual:.2e}, time={solve_time:.2f}s")
    
    if residual > 0.1:
        logger.warning(f"Solver may not have converged, residual={residual:.2e}")
    
    x_block = onp.array(x_gpu)
    solution = extract_complex_solution(x_block, n_orig)
    
    return solution


def solve_hcurl_complex_jax_gpu(A_complex, b_complex, fe, tol=1e-6, maxiter=2000):
    """Solve complex H(curl) system using pure JAX GPU solver.
    
    This uses:
    - Real 2x2 block formulation for complex system
    - JAX sparse BiCGSTAB for GPU-accelerated iteration
    - Jacobi preconditioner (fully on GPU, like original JAX-FEM)
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix (K - k²M with PML).
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    tol : float
        Relative tolerance.
    maxiter : int
        Maximum iterations.
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution.
    """
    from jax.experimental.sparse import BCOO
    import time as time_module
    
    start_time = time_module.time()
    n = A_complex.shape[0]
    
    # Build real 2x2 block system from complex matrix
    logger.info(f"Building 2x2 block real system from {n}x{n} complex matrix...")
    A_block, b_block, n_orig = build_block_complex_system(A_complex, b_complex)
    
    # Convert to JAX BCOO sparse format (this goes to GPU)
    logger.info("Converting to JAX BCOO sparse format for GPU...")
    A_csr = A_block.tocsr()
    A_bcoo = BCOO.from_scipy_sparse(A_csr).sort_indices()
    
    # Transfer data to GPU
    b_gpu = np.array(b_block)
    
    # Jacobi preconditioner (diagonal scaling) - fully on GPU
    jacobi = np.array(A_csr.diagonal())
    jacobi = np.where(np.abs(jacobi) > 1e-14, jacobi, 1.0)
    
    def precond(x):
        return x / jacobi
    
    n_block = 2 * n_orig
    logger.info(f"Running JAX GPU BiCGSTAB on {n_block}x{n_block} sparse block system...")
    logger.info(f"Matrix nnz: {A_csr.nnz}, density: {A_csr.nnz / (n_block**2) * 100:.2f}%")
    
    # JAX sparse BiCGSTAB - this runs fully on GPU!
    x_gpu, info = jax.scipy.sparse.linalg.bicgstab(
        A_bcoo, 
        b_gpu,
        x0=None,
        M=precond,
        tol=tol,
        atol=tol,
        maxiter=maxiter
    )
    x_gpu.block_until_ready()
    
    # Check residual
    residual = np.linalg.norm(A_bcoo @ x_gpu - b_gpu)
    solve_time = time_module.time() - start_time
    logger.info(f"JAX GPU BiCGSTAB: residual={residual:.2e}, time={solve_time:.2f}s")
    
    if residual > 0.1:
        logger.warning(f"Solver may not have converged, residual={residual:.2e}")
    
    # Extract complex solution
    x_block = onp.array(x_gpu)
    solution = extract_complex_solution(x_block, n_orig)
    
    return solution


def petsc_block_ams_solve(A_complex, b_complex, fe, ksp_type='gmres', rtol=1e-8, max_iter=500, pc_type='ilu'):
    """Solve complex H(curl) system using block formulation with PETSc.
    
    Converts complex system to real 2x2 block system and uses iterative solver.
    
    Parameters
    ----------
    A_complex : sparse matrix (complex)
        The complex system matrix (e.g., K - k²M with PML).
    b_complex : numpy.ndarray (complex)
        The complex RHS vector.
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    ksp_type : str
        Krylov solver type (default 'gmres').
    rtol : float
        Relative tolerance.
    max_iter : int
        Maximum iterations.
    pc_type : str
        Preconditioner type: 'ilu', 'ams', 'boomeramg' (default 'ilu').
    
    Returns
    -------
    x_complex : numpy.ndarray (complex)
        The complex solution.
    """
    from petsc4py import PETSc
    from scipy.sparse import bmat
    import time as time_module
    
    start_time = time_module.time()
    
    # Build block system
    logger.info("Building 2x2 block real system from complex matrix...")
    A_block, b_block, n = build_block_complex_system(A_complex, b_complex)
    
    # Convert to PETSc
    logger.info("Converting to PETSc format...")
    A_petsc = PETSc.Mat().createAIJ(
        size=A_block.shape,
        csr=(A_block.indptr.astype(PETSc.IntType),
             A_block.indices.astype(PETSc.IntType),
             A_block.data)
    )
    
    # Setup KSP solver
    logger.info(f"Setting up PETSc {ksp_type.upper()} with {pc_type.upper()} preconditioner...")
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol, max_it=max_iter)
    
    # Setup preconditioner
    pc = ksp.getPC()
    
    if pc_type == 'ams':
        # Build preconditioner matrix (absolute value form)
        P_diag = build_block_preconditioner_matrix(A_complex)
        P_block = bmat([[P_diag, None], 
                        [None, P_diag]], format='csr')
        P_petsc = PETSc.Mat().createAIJ(
            size=P_block.shape,
            csr=(P_block.indptr.astype(PETSc.IntType),
                 P_block.indices.astype(PETSc.IntType),
                 P_block.data)
        )
        ksp.setOperators(A_petsc, P_petsc)
        
        # Discrete gradient for AMS
        G_scipy = compute_discrete_gradient(fe)
        G_block = bmat([[G_scipy, None], [None, G_scipy]], format='csr')
        G_petsc = PETSc.Mat().createAIJ(
            size=G_block.shape,
            csr=(G_block.indptr.astype(PETSc.IntType),
                 G_block.indices.astype(PETSc.IntType),
                 G_block.data)
        )
        
        pc.setType('hypre')
        pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G_petsc)
        
        coords = compute_node_coordinates(fe)
        coords_block = onp.vstack([coords, coords])
        pc.setCoordinates(coords_block)
    elif pc_type == 'boomeramg':
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')
    else:  # Default to ILU
        pc.setType('ilu')
        pc.setFactorLevels(2)  # ILU(2) for better convergence
    
    ksp.setFromOptions()
    
    # Solve
    rhs = PETSc.Vec().createSeq(len(b_block))
    rhs.setValues(range(len(b_block)), b_block)
    x = PETSc.Vec().createSeq(len(b_block))
    
    logger.info("Solving block system...")
    ksp.solve(rhs, x)
    
    # Check convergence
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    
    solve_time = time_module.time() - start_time
    if reason > 0:
        logger.info(f"Block AMS solver converged in {its} iterations, {solve_time:.2f}s")
    else:
        import warnings
        warnings.warn(f"Block AMS solver did not converge: reason={reason}, iterations={its}")
    
    # Extract complex solution
    x_block = x.getArray()
    x_complex = extract_complex_solution(x_block, n)
    
    return x_complex


def compute_dipole_source_rhs(fe, source_point, polarization, amplitude=1.0, sigma=0.05, use_gpu=True):
    """Compute RHS for a dipole source excitation using JAX GPU.
    
    A Hertzian dipole at x_s with polarization p is represented as:
        J = J_0 * p * δ(x - x_s)
    
    This is approximated by a Gaussian-smoothed source:
        J ≈ J_0 * p * exp(-|x - x_s|² / (2σ²))
    
    The RHS is: b_i = ∫ J · N_i dΩ
    
    Parameters
    ----------
    fe : HCurlFiniteElement
        The H(curl) finite element space.
    source_point : array-like
        Location of the dipole source (x, y) or (x, y, z).
    polarization : array-like
        Polarization direction of the dipole.
    amplitude : float
        Source amplitude.
    sigma : float
        Gaussian smoothing width (should be ~2-3 element sizes).
    use_gpu : bool
        Whether to use JAX GPU acceleration (default True).
    
    Returns
    -------
    b : numpy.ndarray (complex)
        The RHS vector.
    """
    import basix
    import time as time_module
    
    start_time = time_module.time()
    num_dofs = fe.num_total_dofs
    num_cells = fe.num_cells
    num_quads = fe.num_quads
    dim = fe.dim
    
    source_point = onp.array(source_point[:dim])
    polarization = onp.array(polarization[:dim])
    polarization = polarization / onp.linalg.norm(polarization)
    
    # Get reference quadrature points
    if dim == 2:
        cell_type = basix.CellType.triangle
    else:
        cell_type = basix.CellType.tetrahedron
    
    quad_points_ref, _ = basix.make_quadrature(cell_type, fe.gauss_order)
    
    # Compute physical quadrature points for all cells
    cell_coords = fe.points[fe.cells]  # (num_cells, nodes_per_cell, dim)
    
    if dim == 2:
        # Triangle: x = (1-ξ-η)*x0 + ξ*x1 + η*x2
        xi = quad_points_ref[:, 0]   # (num_quads,)
        eta = quad_points_ref[:, 1]  # (num_quads,)
        # Barycentric weights: (num_quads, 3)
        bary = onp.stack([1 - xi - eta, xi, eta], axis=1)
        # Physical quad points: (num_cells, num_quads, dim)
        phys_quads = onp.einsum('cn d,qn->cqd', cell_coords, bary)
    else:
        xi = quad_points_ref[:, 0]
        eta = quad_points_ref[:, 1]
        zeta = quad_points_ref[:, 2]
        bary = onp.stack([1 - xi - eta - zeta, xi, eta, zeta], axis=1)
        phys_quads = onp.einsum('cnd,qn->cqd', cell_coords, bary)
    
    if use_gpu and jax.devices()[0].platform == 'gpu':
        logger.info("Using JAX GPU for dipole source RHS")
        
        phys_quads_j = np.array(phys_quads)
        source_point_j = np.array(source_point)
        polarization_j = np.array(polarization)
        shape_vals_j = np.array(fe.shape_vals_physical)
        JxW_j = np.array(fe.JxW)
        
        @jax.jit
        def compute_local_rhs(phys_quads, source_point, polarization, shape_vals, JxW, amp, sig):
            # phys_quads: (cells, quads, dim)
            # shape_vals: (cells, quads, dofs, dim)
            
            # Distance squared from source: (cells, quads)
            r2 = np.sum((phys_quads - source_point[None, None, :]) ** 2, axis=-1)
            
            # Gaussian source value: (cells, quads)
            source_vals = amp * np.exp(-r2 / (2 * sig ** 2))
            
            # Dot product of polarization with basis functions: (cells, quads, dofs)
            pol_dot_N = np.einsum('d,cqid->cqi', polarization, shape_vals)
            
            # Local RHS: sum over quads
            b_local = np.einsum('cq,cqi,cq->ci', source_vals, pol_dot_N, JxW)
            
            return b_local
        
        b_local = onp.array(compute_local_rhs(
            phys_quads_j, source_point_j, polarization_j, 
            shape_vals_j, JxW_j, amplitude, sigma
        ))
        
        gpu_time = time_module.time() - start_time
        logger.info(f"GPU dipole RHS computation: {gpu_time:.4f} s")
    else:
        logger.info("Using NumPy CPU for dipole source RHS")
        
        # Distance squared: (cells, quads)
        r2 = onp.sum((phys_quads - source_point[None, None, :]) ** 2, axis=-1)
        source_vals = amplitude * onp.exp(-r2 / (2 * sigma ** 2))
        
        # Dot product: (cells, quads, dofs)
        pol_dot_N = onp.einsum('d,cqid->cqi', polarization, fe.shape_vals_physical)
        
        # Local RHS
        b_local = onp.einsum('cq,cqi,cq->ci', source_vals, pol_dot_N, fe.JxW)
    
    # Vectorized global RHS assembly using numpy.add.at
    b = onp.zeros(num_dofs, dtype=complex)
    cell_dofs = onp.array(fe.cell_to_dofs)  # (num_cells, dofs_per_cell)
    onp.add.at(b, cell_dofs.ravel(), b_local.ravel())
    
    total_time = time_module.time() - start_time
    logger.info(f"Dipole source RHS total: {total_time:.4f} s")
    
    return b
