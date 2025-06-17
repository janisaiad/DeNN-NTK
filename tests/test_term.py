import pytest
import jax.numpy as jnp
import sympy as sp
from finitewidth import Term

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






def test_term_replace_rules():
    """Test replace_rules method of Term class"""
    # Create sample term
    entry_vectors = jnp.ones((3, 2))
    H = 2
    term = Term(entry_vectors, H)
    
    # Test invalid rule type
    with pytest.raises(ValueError):
        term.replace_rules({'invalid_rule': {}})
        
    # Test valid rule types
    rules = {
        'a_t': {},
        'W_forward': {},
        'W_backward': {},
        'x_layer': {},
        'sigma_prime_i': {}
    }
    result = term.replace_rules(rules)
    assert isinstance(result, sp.Expr)

if __name__ == "__main__":
    test_term_init()
    test_term_replace_rules()