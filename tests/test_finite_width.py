import numpy as np
import jax.numpy as jnp
from jax.nn import relu
import pytest

from model.finitewidth.finitewidth_ntk import NtkEmpiricalJax, NtkEmpiricalNeuralTangent
from model.finitewidth.kernel3_mean import Kernel3Mean
from model.finitewidth.kernel3_empirical import Kernel3Empirical
from model.finitewidth.finitewidth_formal import FormalKernel, Term

def test_ntk_empirical_jax():
    """Test NtkEmpiricalJax class"""
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
    
    ntk = NtkEmpiricalJax(weights, sigma_derivatives, feature_maps)
    
    # Test full NTK computation
    ntk_val, layer_ntks = ntk.ntk(0, 1)
    assert isinstance(ntk_val, float)
    assert isinstance(layer_ntks, dict)
    assert len(layer_ntks) == H + 1
    
    # Test layer NTK computation
    layer_ntk = ntk.layer_ntk(1, 0, 1)
    assert isinstance(layer_ntk, float)

def test_ntk_empirical_neural_tangent():
    """Test NtkEmpiricalNeuralTangent class"""
    m = 10
    H = 2
    d = 3
    n = 2
    
    weights = [
        jnp.ones((m, d)),
        jnp.ones((m, m)), 
        jnp.ones(m)
    ]
    
    feature_maps = {
        0: [jnp.ones(d) for _ in range(n)]
    }
    
    ntk = NtkEmpiricalNeuralTangent(weights, feature_maps, relu)
    
    # Test full NTK computation
    ntk_val = ntk.ntk(0, 1)
    assert isinstance(ntk_val, float)

def test_kernel3_mean():
    """Test Kernel3Mean class"""
    m = 10
    H = 2
    d = 3
    n = 2
    
    weights = [
        jnp.ones((m, d)),
        jnp.ones((m, m)),
        jnp.ones(m)
    ]
    
    sigma_derivatives = {
        i: [jnp.eye(m) for _ in range(n)]
        for i in range(H + 1)
    }
    
    feature_maps = {
        i: [jnp.ones(m if i > 0 else d) for _ in range(n)]
        for i in range(H + 1)
    }
    
    k3 = Kernel3Mean(weights, sigma_derivatives, feature_maps)
    
    # Test kernel3 computation
    k3_val = k3.kernel3(0, 1, 0)
    assert isinstance(k3_val, float)
    
    # Test helper functions
    G = k3._compute_G(1, 0)
    assert G.shape == (m,)
    
    delta_x = k3._compute_delta_x(1, 0, 1)
    assert delta_x.shape == (m,)
    
    delta_G = k3._compute_delta_G(1, 0, 1)
    assert delta_G.shape == (m,)

def test_kernel3_empirical():
    """Test Kernel3Empirical class"""
    m = 10
    H = 2
    d = 3
    n = 2
    
    weights = [
        jnp.ones((m, d)),
        jnp.ones((m, m)),
        jnp.ones(m)
    ]
    
    sigma_derivatives = {
        i: [jnp.eye(m) for _ in range(n)]
        for i in range(H + 1)
    }
    
    feature_maps = {
        i: [jnp.ones(m if i > 0 else d) for _ in range(n)]
        for i in range(H + 1)
    }
    
    k3 = Kernel3Empirical(weights, sigma_derivatives, feature_maps)
    
    # Test kernel3 computation
    k3_val = k3.kernel3(0, 1, 0)
    assert isinstance(k3_val, float)
    
    # Test helper functions
    G = k3._compute_G(1, 0)
    assert G.shape == (m,)
    
    delta_x = k3._compute_delta_x(1, 0, 1)
    assert delta_x.shape == (m,)
    
    delta_G = k3._compute_delta_G(1, 0, 1)
    assert delta_G.shape == (m,)

def test_formal_kernel():
    """Test FormalKernel class"""
    n_entries = 2
    dim_input = 3
    entry_vectors = jnp.array([[1., 0., 0.],
                              [0., 1., 0.]]).T  # Shape (dim_input, n_entries)
    H = 2
    
    kernel = FormalKernel(n_entries, dim_input, entry_vectors, H)
    
    assert kernel.n_entries == n_entries
    assert kernel.dim_input == dim_input
    assert kernel.H == H
    assert jnp.array_equal(kernel.entry_vectors, entry_vectors)
    
    # Test Term class
    term = Term(entry_vectors, H)
    assert term.H == H
    assert jnp.array_equal(term.entry_vectors, entry_vectors)
    
    # Test evolution
    rules = {
        'a_t': True,
        'W_forward': True,
        'W_backward': True,
        'x_layer': True,
        'sigma_prime_i': True
    }
    evolved_term = term.replace_rules(rules)
    assert isinstance(evolved_term, sp.Expr)
