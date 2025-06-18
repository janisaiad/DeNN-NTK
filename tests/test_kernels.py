import numpy as np
import jax.numpy as jnp
import pytest
from model.infinitewidth.infinitewidth import InfiniteWidth
from model.infinitewidth.kernel3_infinite import Kernel3Infinite
from model.finitewidth.kernel3_empirical import Kernel3Empirical
from model.finitewidth.finitewidth import Kernel

def test_infinite_width_initialization():
    """Test initialization of InfiniteWidth class"""
    iw = InfiniteWidth(n_layers=3, n_outputs=1)
    assert iw.l == 3
    assert iw.ml == 1
    assert iw.a == 1.0
    assert iw.b == 1.0
    assert np.isclose(iw.delta_phi, 0.5)
    assert np.isclose(iw.sigma, 1/np.sqrt(2))

def test_infinite_width_kernel_matrix():
    """Test kernel matrix computation for InfiniteWidth"""
    iw = InfiniteWidth(n_layers=2, n_outputs=1)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = iw.kernel_matrix(X)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T)  # Check symmetry
    assert np.all(np.diag(K) >= 0)  # Check positive diagonal

def test_kernel3_infinite():
    """Test Kernel3Infinite class"""
    k3 = Kernel3Infinite(n_layers=2, n_outputs=1)
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0]) 
    x3 = np.array([1.0, 1.0]) / np.sqrt(2)
    
    k3_val = k3._kernel_3_entry(x1, x2, x3)
    assert isinstance(k3_val, float)

def test_kernel3_empirical():
    """Test Kernel3Empirical class"""
    # Setup mock data
    m = 10  # Width
    H = 2   # Depth
    d = 3   # Input dimension
    n = 2   # Number of samples
    
    # Mock weights
    weights = [
        jnp.ones((m, d)),  # W^1
        jnp.ones((m, m)),  # W^2
        jnp.ones(m)        # a
    ]
    
    # Mock feature maps and derivatives
    sigma_derivatives = {
        i: [jnp.eye(m) for _ in range(n)]
        for i in range(H + 1)
    }
    
    feature_maps = {
        i: [jnp.ones(m if i > 0 else d) for _ in range(n)]
        for i in range(H + 1)
    }
    
    k3 = Kernel3Empirical(weights, sigma_derivatives, feature_maps)
    
    # Test kernel2 computation
    k2_val = k3.kernel2(0, 1)
    assert isinstance(k2_val, float)
    
    # Test kernel3 computation
    k3_val = k3.kernel3(0, 1, 0)
    assert isinstance(k3_val, float)

def test_finite_width_kernel():
    """Test Kernel class from finitewidth module"""
    n_entries = 2
    dim_input = 3
    entry_vectors = jnp.array([[1., 0., 0.],
                              [0., 1., 0.]]).T  # Shape (dim_input, n_entries)
    H = 2
    
    kernel = Kernel(n_entries, dim_input, entry_vectors, H)
    
    assert kernel.n_entries == n_entries
    assert kernel.dim_input == dim_input
    assert kernel.H == H
    assert jnp.array_equal(kernel.entry_vectors, entry_vectors)
