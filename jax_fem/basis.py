import basix
import numpy as onp

from jax_fem import logger


# def get_full_integration_poly_degree(ele_type, lag_order, dim):
#     """Only works for weak forms of (grad_u, grad_v).
#     TODO: Is this correct?
#     Reference:
#     https://zhuanlan.zhihu.com/p/521630645
#     """
#     if ele_type == 'hexahedron' or ele_type == 'quadrilateral':
#         return 2 * (dim*lag_order - 1)

#     if ele_type == 'tetrahedron' or ele_type == 'triangle':
#         return 2 * (dim*(lag_order - 1) - 1)

def get_elements(ele_type):
    """Obtain element information useful for `basix <https://github.com/FEniCS/basix>`_ to handle. 

    Note that mesh node ordering is important.
    If the input mesh file is Gmsh .msh or Abaqus .inp, meshio would convert it to its own ordering.
    Our experience shows that meshio ordering is the same as Abaqus.
    For example, for a 10-node tetrahedron element, the ordering of meshio follows this 
    `instruction <https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html>`_.
    The troublesome thing is that basix has `a different ordering <https://defelement.com/elements/lagrange.html>`_. 
    The consequence is that we need to define the  **re_order** variable to make sure the ordering is correct.


    Parameters
    ----------
    ele_type: str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`

    Returns
    -------
    element_family : BasixObject
        `ElementFamily <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.ElementFamily>`_
    basix_ele : BasixObject
        For element: `CellType <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.CellType>`_
    basix_face_ele : BasixObject
        For element face: `CellType <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.CellType>`_
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`
    degree : int
        Element degree, used in basix
    re_order : list
        Specifieds node re-ordering transformation. For example, [0, 1, 3, 2, 4, 5, 7, 6] for HEX8 element.
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 2x2x2, TODO: is this full integration?
        degree = 1
    elif ele_type == 'HEX27':
        print(f"Warning: 27-node hexahedron is rarely used in practice and not recommended.")
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 10 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
        degree = 2
    # TODO: Check if this is correct.
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 # 3, full integration
        degree = 2
    # ============== Nedelec (first kind) H(curl) elements ==============
    # DOFs are associated with edges, suitable for electromagnetic problems
    elif ele_type == 'N1E_TRI1':  # Nedelec first kind, degree 1, triangle
        re_order = None  # Nedelec elements use edge-based DOFs, no node reordering needed
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'N1E_TRI2':  # Nedelec first kind, degree 2, triangle
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 4
        degree = 2
    elif ele_type == 'N1E_TET1':  # Nedelec first kind, degree 1, tetrahedron
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2
        degree = 1
    elif ele_type == 'N1E_TET2':  # Nedelec first kind, degree 2, tetrahedron
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 4
        degree = 2
    elif ele_type == 'N1E_QUAD1':  # Nedelec first kind, degree 1, quadrilateral
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'N1E_QUAD2':  # Nedelec first kind, degree 2, quadrilateral
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 4
        degree = 2
    elif ele_type == 'N1E_HEX1':  # Nedelec first kind, degree 1, hexahedron
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2
        degree = 1
    elif ele_type == 'N1E_HEX2':  # Nedelec first kind, degree 2, hexahedron
        re_order = None
        element_family = basix.ElementFamily.N1E
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 4
        degree = 2
    # ============== Nedelec (second kind) H(curl) elements ==============
    elif ele_type == 'N2E_TRI1':  # Nedelec second kind, degree 1, triangle
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'N2E_TRI2':  # Nedelec second kind, degree 2, triangle
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 4
        degree = 2
    elif ele_type == 'N2E_TET1':  # Nedelec second kind, degree 1, tetrahedron
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2
        degree = 1
    elif ele_type == 'N2E_TET2':  # Nedelec second kind, degree 2, tetrahedron
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 4
        degree = 2
    elif ele_type == 'N2E_QUAD1':  # Nedelec second kind, degree 1, quadrilateral
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'N2E_QUAD2':  # Nedelec second kind, degree 2, quadrilateral
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 4
        degree = 2
    elif ele_type == 'N2E_HEX1':  # Nedelec second kind, degree 1, hexahedron
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2
        degree = 1
    elif ele_type == 'N2E_HEX2':  # Nedelec second kind, degree 2, hexahedron
        re_order = None
        element_family = basix.ElementFamily.N2E
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 4
        degree = 2
    else:
        raise NotImplementedError(f"Element type '{ele_type}' is not implemented. "
                                  f"Supported types: HEX8, HEX20, HEX27, TET4, TET10, QUAD4, QUAD8, TRI3, TRI6, "
                                  f"N1E_TRI1, N1E_TRI2, N1E_TET1, N1E_TET2, N1E_QUAD1, N1E_QUAD2, N1E_HEX1, N1E_HEX2, "
                                  f"N2E_TRI1, N2E_TRI2, N2E_TET1, N2E_TET2, N2E_QUAD1, N2E_QUAD2, N2E_HEX1, N2E_HEX2")

    return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order


def is_hcurl_element(ele_type):
    """Check if the element type is an H(curl) conforming element (Nedelec).
    
    Parameters
    ----------
    ele_type : str
        Element type string
        
    Returns
    -------
    bool
        True if the element is an H(curl) element (Nedelec first or second kind)
    """
    return ele_type.startswith('N1E_') or ele_type.startswith('N2E_')


def reorder_inds(inds, re_order):
    """Apply re-ordering transformation for node indices.
    """
    if re_order is None:
        return inds
    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type, gauss_order=None):
    """Use `basix <https://github.com/FEniCS/basix>`_ to get shape function values and gradients for elements.

    Parameters
    ----------
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`

    Returns
    -------
    For scalar Lagrange elements:
        shape_values: NumpyArray
            Shape is (num_quads, num_nodes), e.g, (8, 8) for HEX8 element.
        shape_grads_ref: NumpyArray
            Shape is (num_quads, num_nodes, dim), e.g, (8, 8, 3) for HEX8 element.
        weights: NumpyArray
            Shape is (num_quads,), e.g, (8,) for HEX8 element.
    
    For H(curl) Nedelec elements:
        shape_values: NumpyArray
            Shape is (num_quads, num_dofs, dim), e.g, (3, 3, 2) for N1E_TRI1 element.
            These are vector-valued basis functions.
        shape_grads_ref: NumpyArray
            Shape is (num_quads, num_dofs, dim, dim) for derivatives.
        weights: NumpyArray
            Shape is (num_quads,), e.g, (3,) for N1E_TRI1 element.
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    element = basix.create_element(element_family, basix_ele, degree)
    
    if is_hcurl_element(ele_type):
        # H(curl) elements have vector-valued basis functions
        # tabulate returns shape: (num_derivatives, num_points, num_dofs, value_size)
        # For H(curl) in 2D: value_size = 2, in 3D: value_size = 3
        vals_and_grads = element.tabulate(1, quad_points)
        # vals_and_grads[0] is the function values: (num_points, num_dofs, dim)
        shape_values = vals_and_grads[0]  # (num_quads, num_dofs, dim)
        
        # For gradients, we need derivatives w.r.t each spatial direction
        # vals_and_grads[1:dim+1] contains derivatives
        dim = quad_points.shape[1]
        # shape_grads_ref: (num_quads, num_dofs, dim, dim) 
        # [i, j, k, l] = d(basis_j component k)/d(x_l) at quad point i
        shape_grads_ref = onp.stack([vals_and_grads[i+1] for i in range(dim)], axis=-1)
        
        logger.debug(f"ele_type = {ele_type} (H(curl)), quad_points.shape = {quad_points.shape}, "
                     f"shape_values.shape = {shape_values.shape}, shape_grads_ref.shape = {shape_grads_ref.shape}")
    else:
        # Standard scalar Lagrange elements
        vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
        shape_values = vals_and_grads[0, :, :, 0]
        shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
        logger.debug(f"ele_type = {ele_type}, quad_points.shape = (num_quads, dim) = {quad_points.shape}")
    
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type, gauss_order=None):
    """Use `basix <https://github.com/FEniCS/basix>`_ to get shape function values and gradients for element faces.

    Parameters
    ----------
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`

    Returns
    -------
    face_shape_vals: NumpyArray
        Shape is (num_faces, num_face_quads, num_nodes), e.g, (6, 4, 8) for HEX8 element.
    face_shape_grads_ref: NumpyArray
        Shape is(num_faces, num_face_quads, num_nodes, dim), e.g, (6, 4, 3) for HEX8 element.
    face_weights: NumpyArray
        Shape is (num_faces, num_face_quads), e.g, (6, 4) for HEX8 element.
    face_normals:NumpyArray
        Shape is (num_faces, dim), e.g, (6, 3) for HEX8 element.
    face_inds: NumpyArray
        Shape is (num_faces, num_face_vertices), e.g, (6, 4) for HEX8 element.
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    # TODO: Check if this is correct.
    # We should provide freedom for seperate gauss_order for volume integral and surface integral
    # Currently, they're using the same gauss_order!
    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)
    
    if is_hcurl_element(ele_type):
        # H(curl) elements have vector-valued basis functions
        vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))
        # face_shape_vals: (num_faces, num_face_quads, num_dofs, dim)
        face_shape_vals = vals_and_grads[0].reshape(num_faces, num_face_quads, -1, dim)
        # face_shape_grads_ref: (num_faces, num_face_quads, num_dofs, dim, dim)
        face_shape_grads_ref = onp.stack([vals_and_grads[i+1] for i in range(dim)], axis=-1)
        face_shape_grads_ref = face_shape_grads_ref.reshape(num_faces, num_face_quads, -1, dim, dim)
    else:
        vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
        face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
        face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
        face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    
    logger.debug(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds
