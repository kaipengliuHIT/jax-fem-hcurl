"""
Tests for H(curl) conforming Nedelec elements.

This module tests the implementation of Nedelec edge elements for electromagnetic problems.
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jax_fem.basis import get_elements, get_shape_vals_and_grads, is_hcurl_element
from jax_fem.generate_mesh import Mesh, box_mesh


class TestNedelecBasis:
    """Test Nedelec basis functions from basix."""
    
    def test_is_hcurl_element(self):
        """Test element type detection."""
        # H(curl) elements
        assert is_hcurl_element('N1E_TRI1') == True
        assert is_hcurl_element('N1E_TRI2') == True
        assert is_hcurl_element('N1E_TET1') == True
        assert is_hcurl_element('N2E_HEX1') == True
        
        # Lagrange elements
        assert is_hcurl_element('TRI3') == False
        assert is_hcurl_element('HEX8') == False
        assert is_hcurl_element('TET4') == False
    
    def test_get_elements_n1e_tri1(self):
        """Test element info for N1E_TRI1."""
        import basix
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = \
            get_elements('N1E_TRI1')
        
        assert element_family == basix.ElementFamily.N1E
        assert basix_ele == basix.CellType.triangle
        assert basix_face_ele == basix.CellType.interval
        assert degree == 1
        assert re_order is None  # Nedelec elements don't use node reordering
    
    def test_get_elements_n1e_tet1(self):
        """Test element info for N1E_TET1."""
        import basix
        element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = \
            get_elements('N1E_TET1')
        
        assert element_family == basix.ElementFamily.N1E
        assert basix_ele == basix.CellType.tetrahedron
        assert basix_face_ele == basix.CellType.triangle
        assert degree == 1
    
    def test_shape_vals_n1e_tri1(self):
        """Test shape function values for N1E_TRI1."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('N1E_TRI1')
        
        # N1E_TRI1 has 3 DOFs (one per edge) and is vector-valued (dim=2)
        assert shape_vals.ndim == 3
        assert shape_vals.shape[1] == 3  # 3 DOFs
        assert shape_vals.shape[2] == 2  # 2D vectors
        
        # Gradients should be (num_quads, num_dofs, dim, dim)
        assert shape_grads_ref.ndim == 4
        assert shape_grads_ref.shape[1] == 3
        assert shape_grads_ref.shape[2] == 2
        assert shape_grads_ref.shape[3] == 2
    
    def test_shape_vals_n1e_tet1(self):
        """Test shape function values for N1E_TET1."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('N1E_TET1')
        
        # N1E_TET1 has 6 DOFs (one per edge) and is vector-valued (dim=3)
        assert shape_vals.ndim == 3
        assert shape_vals.shape[1] == 6  # 6 DOFs (6 edges in tetrahedron)
        assert shape_vals.shape[2] == 3  # 3D vectors
        
        # Gradients
        assert shape_grads_ref.ndim == 4
        assert shape_grads_ref.shape[1] == 6
        assert shape_grads_ref.shape[2] == 3
        assert shape_grads_ref.shape[3] == 3

    def test_shape_vals_n2e_tri1(self):
        """Test shape function values for Nedelec second kind N2E_TRI1."""
        shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads('N2E_TRI1')
        
        # N2E_TRI1 (second kind, degree 1) has more DOFs than N1E_TRI1
        assert shape_vals.ndim == 3
        assert shape_vals.shape[2] == 2  # 2D vectors
    
    def test_quadrature_weights_positive(self):
        """Test that quadrature weights are positive."""
        for ele_type in ['N1E_TRI1', 'N1E_TET1', 'N1E_QUAD1', 'N1E_HEX1']:
            _, _, weights = get_shape_vals_and_grads(ele_type)
            assert np.all(weights > 0), f"Negative weights for {ele_type}"


class TestHCurlFiniteElement:
    """Test the HCurlFiniteElement class."""
    
    @pytest.fixture
    def simple_triangle_mesh(self):
        """Create a simple single triangle mesh."""
        points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        cells = np.array([[0, 1, 2]])
        return Mesh(points, cells)
    
    @pytest.fixture
    def unit_square_mesh(self):
        """Create a unit square mesh with 2 triangles."""
        points = np.array([
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]
        ])
        cells = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        return Mesh(points, cells)
    
    def test_hcurl_fe_creation(self, simple_triangle_mesh):
        """Test HCurlFiniteElement instantiation."""
        from jax_fem.hcurl_fe import HCurlFiniteElement
        
        fe = HCurlFiniteElement(
            mesh=simple_triangle_mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        assert fe.num_cells == 1
        assert fe.num_dofs_per_cell == 3
        assert fe.num_total_edges == 3
        assert fe.dim == 2
    
    def test_edge_dof_map(self, unit_square_mesh):
        """Test edge to DOF mapping."""
        from jax_fem.hcurl_fe import HCurlFiniteElement
        
        fe = HCurlFiniteElement(
            mesh=unit_square_mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        # 2 triangles share one edge, so 5 unique edges
        assert fe.num_total_edges == 5
        assert fe.cell_to_dofs.shape == (2, 3)
    
    def test_physical_quad_points(self, simple_triangle_mesh):
        """Test physical quadrature point computation."""
        from jax_fem.hcurl_fe import HCurlFiniteElement
        
        fe = HCurlFiniteElement(
            mesh=simple_triangle_mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        quad_points = fe.get_physical_quad_points()
        
        # All quad points should be inside the triangle
        assert quad_points.shape[0] == 1  # 1 cell
        assert quad_points.shape[2] == 2  # 2D
        
        # Points should be in [0,1] x [0,1] and x+y <= 1
        for pt in quad_points[0]:
            assert pt[0] >= 0 and pt[0] <= 1
            assert pt[1] >= 0 and pt[1] <= 1
            assert pt[0] + pt[1] <= 1 + 1e-10


class TestCurlCurlMatrix:
    """Test curl-curl stiffness matrix assembly."""
    
    def test_curl_curl_symmetry(self):
        """Test that curl-curl matrix is symmetric."""
        from jax_fem.hcurl_fe import HCurlFiniteElement, compute_curl_curl_matrix
        
        points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        cells = np.array([[0, 1, 2]])
        mesh = Mesh(points, cells)
        
        fe = HCurlFiniteElement(
            mesh=mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        K = compute_curl_curl_matrix(fe)
        
        # Check symmetry
        npt.assert_array_almost_equal(K.toarray(), K.T.toarray(), decimal=10)
    
    def test_curl_curl_positive_semidefinite(self):
        """Test that curl-curl matrix is positive semi-definite."""
        from jax_fem.hcurl_fe import HCurlFiniteElement, compute_curl_curl_matrix
        
        points = np.array([
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]
        ])
        cells = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        mesh = Mesh(points, cells)
        
        fe = HCurlFiniteElement(
            mesh=mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        K = compute_curl_curl_matrix(fe)
        eigenvalues = np.linalg.eigvalsh(K.toarray())
        
        # All eigenvalues should be >= 0 (with numerical tolerance)
        assert np.all(eigenvalues >= -1e-10)


class TestMassMatrix:
    """Test mass matrix assembly."""
    
    def test_mass_symmetry(self):
        """Test that mass matrix is symmetric."""
        from jax_fem.hcurl_fe import HCurlFiniteElement, compute_mass_matrix
        
        points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        cells = np.array([[0, 1, 2]])
        mesh = Mesh(points, cells)
        
        fe = HCurlFiniteElement(
            mesh=mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        M = compute_mass_matrix(fe)
        
        # Check symmetry
        npt.assert_array_almost_equal(M.toarray(), M.T.toarray(), decimal=10)
    
    def test_mass_positive_definite(self):
        """Test that mass matrix is positive definite."""
        from jax_fem.hcurl_fe import HCurlFiniteElement, compute_mass_matrix
        
        points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        cells = np.array([[0, 1, 2]])
        mesh = Mesh(points, cells)
        
        fe = HCurlFiniteElement(
            mesh=mesh,
            dim=2,
            ele_type='N1E_TRI1',
            gauss_order=2,
            dirichlet_bc_info=None
        )
        
        M = compute_mass_matrix(fe)
        eigenvalues = np.linalg.eigvalsh(M.toarray())
        
        # All eigenvalues should be > 0
        assert np.all(eigenvalues > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
