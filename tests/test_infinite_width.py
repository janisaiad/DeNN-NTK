import pytest
import numpy as np
from infinitewidth import InfiniteWidth


def test_import():
    """Test if the InfiniteWidth class can be imported"""
    from infinitewidth import InfiniteWidth
    assert InfiniteWidth is not None

def test_infinite_width_init():
    """Test initialization of InfiniteWidth class"""
    n_layers = 3
    n_outputs = 1
    a = 1.0
    b = 1.0
    
    model = InfiniteWidth(n_layers, n_outputs, a, b)
    
    assert model.l == n_layers
    assert model.ml == n_outputs
    assert model.a == a
    assert model.b == b
    assert model.delta_phi == b**2 / (a**2 + b**2)
    assert model.sigma == (a**2 + b**2)**-0.5

def test_varrho():
    """Test the cosine map function"""
    model = InfiniteWidth(3, 1)
    rho = np.array([0.5])
    result = model._varrho(rho)
    assert isinstance(result, np.ndarray)
    
def test_kernel_matrix():
    """Test kernel matrix computation"""
    model = InfiniteWidth(3, 1)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = model.kernel_matrix(X)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T)  # Check symmetry

def test_invalid_parameters():
    """Test initialization with invalid parameters"""
    with pytest.raises(ValueError):
        InfiniteWidth(3, 1, a=0.0, b=0.0)

if __name__ == "__main__":
    pytest.main([__file__])