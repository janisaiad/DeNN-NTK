import pytest
import sympy as sp
from finitewidth import FormalExpression

def test_formal_expression_init():
    """Test initialization of FormalExpression class"""
    expr = FormalExpression()
    
    # Test basic symbols
    assert isinstance(expr.t, sp.Symbol)
    assert str(expr.t) == 't'
    assert isinstance(expr.m, sp.Symbol)
    assert str(expr.m) == 'm'
    
    # Test matrix symbols for each layer
    for layer in range(101):  # MAX_LAYERS + 1 = 101
        assert isinstance(expr.W_symbols[str(layer)], sp.MatrixSymbol)
        assert expr.W_symbols[str(layer)].shape == (expr.m, expr.m)
        assert str(expr.W_symbols[str(layer)]) == 'W'
    
    # Test x symbols for each hierarchy level and layer
    for i in range(10):  # MAX_HIERARCHY_DEPTH = 10
        for layer in range(101):
            key = f"{i}_{layer}"
            assert isinstance(expr.x_symbols[key], sp.Symbol)
            assert str(expr.x_symbols[key]) == f'x_{i}^{layer}'
    
    # Test sigma prime symbols for each hierarchy level and layer
    for r in range(10):  # MAX_HIERARCHY_DEPTH = 10
        for layer in range(101):
            key = f"{r}_{layer}"
            assert isinstance(expr.sigma_prime_symbols[key], sp.FunctionClass)
            assert str(expr.sigma_prime_symbols[key]) == f'σ_{layer}^({r})'


if __name__ == "__main__":
    test_formal_expression_init()