import pytest
import jax.numpy as jnp
import sympy as sp
from finitewidth.finitewidth import Term

def test_term_init():
    """Test initialization of Term class"""
    # Create sample input vectors
    dim_input = 3
    n_entries = 2
    H = 2
    entry_vectors = jnp.ones((dim_input, n_entries))
    
    # Initialize Term
    term = Term(entry_vectors, H)
    
    # Test attributes
    assert isinstance(term.entry_vectors, jnp.ndarray)
    assert term.entry_vectors.shape == (dim_input, n_entries)
    assert term.index_hierarchy == n_entries
    assert term.H == H
    
    # Test symbolic term initialization
    assert isinstance(term.symbolic_term, sp.Expr)
    assert term.expr is not None
    
    # Test that symbolic term is initialized correctly
    expected_term = term.expr.a.T * term.expr.x_symbols['0_' + str(H)]
    assert term.symbolic_term == expected_term

def test_term_replace_rules_single():
    """Test replace_rules method with a single rule."""
    entry_vectors = jnp.ones((3, 2))
    H = 2
    term = Term(entry_vectors, H)
    
    # Test applying 'a_t' rule
    rules_to_apply = {'a_t': True}
    result = term.replace_rules(rules_to_apply)
    
    # Define expected expression
    expr = term.expr
    x_beta_H_minus_1 = sp.MatrixSymbol(f'x_beta^{H-1}', expr.m, 1)
    expected_expr = x_beta_H_minus_1.T * expr.x_symbols['0_' + str(H)]
    
    assert result == expected_expr
    # The original term should be unchanged
    assert term.symbolic_term != result

def test_term_replace_rules_all():
    """Test replace_rules method with all rules."""
    entry_vectors = jnp.ones((3, 2))
    H = 2
    term = Term(entry_vectors, H)
    
    # Test applying all rules
    all_rules = {
        'a_t': True,
        'W_forward': True,
        'W_backward': True,
        'x_layer': True,
        'sigma_prime_i': True
    }
    result = term.replace_rules(all_rules)
    assert isinstance(result, sp.Expr)
    
    # Check that the main symbol 'a' has been replaced.
    assert not result.has(term.expr.a)
    
def test_evolve():
    """Test the evolve method."""
    dim_input = 3
    n_entries = 2
    H = 2
    entry_vectors = jnp.ones((dim_input, n_entries))
    term = Term(entry_vectors, H)
    
    x_beta = jnp.zeros((dim_input, 1))
    new_term = term.evolve(x_beta)
    
    # Check the new term's properties
    assert isinstance(new_term, Term)
    assert new_term.index_hierarchy == n_entries + 1
    assert new_term.entry_vectors.shape == (dim_input, n_entries + 1)
    
    # Check that the symbolic expression has been evolved
    assert isinstance(new_term.symbolic_term, sp.Expr)
    # The evolved term should be different from the original one
    assert new_term.symbolic_term != term.symbolic_term
    # The evolution replaces 'a', so it should not be present in the new term.
    assert not new_term.symbolic_term.has(term.expr.a)

if __name__ == "__main__":
    pytest.main([__file__])