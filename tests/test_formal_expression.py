import pytest
import sympy as sp
import jax.numpy as jnp
from finitewidth import FormalExpression, MAX_HIERARCHY_DEPTH, MAX_LAYERS

def test_formal_expression_init():
    """Test initialization of FormalExpression class"""
    expr = FormalExpression()
    
    # Test base symbols
    assert isinstance(expr.t, sp.Symbol)
    assert str(expr.t) == 't'
    assert isinstance(expr.m, sp.Symbol) 
    assert str(expr.m) == 'm'
    assert isinstance(expr.H, sp.Symbol)
    assert str(expr.H) == 'H'
    
    # Test matrix symbols
    assert isinstance(expr.W, sp.MatrixSymbol)
    assert expr.W.shape == (expr.m, expr.m)
    assert isinstance(expr.a, sp.MatrixSymbol)
    assert expr.a.shape == (expr.m, 1)

    # Test x symbols dictionary
    for i in range(MAX_HIERARCHY_DEPTH):
        for layer in range(MAX_LAYERS+1):
            key = f"{i}_{layer}"
            assert key in expr.x_symbols
            assert isinstance(expr.x_symbols[key], sp.Symbol)
            assert str(expr.x_symbols[key]) == f"x_{i}^{layer}"

    # Test sigma prime symbols dictionary  
    for r in range(MAX_HIERARCHY_DEPTH):
        assert str(r) in expr.sigma_prime_symbols
        assert isinstance(expr.sigma_prime_symbols[str(r)], sp.FunctionClass)
        assert str(expr.sigma_prime_symbols[str(r)]) == f"σ^({r})"
