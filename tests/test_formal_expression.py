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
    
    # Test matrix symbols
    assert isinstance(expr.W, sp.MatrixSymbol)
    assert expr.W.shape == (expr.m, expr.m)
    
    # Test x symbol
    assert isinstance(expr.x_symbols['0_0'], sp.Symbol)
    assert str(expr.x_symbols['0_0']) == 'x_0^0'
    
    # Test sigma prime symbol
    assert isinstance(expr.sigma_prime_symbols['0'], sp.FunctionClass)
    assert str(expr.sigma_prime_symbols['0']) == 'σ^(0)'


if __name__ == "__main__":
    test_formal_expression_init()