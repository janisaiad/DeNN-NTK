import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Literal
import sympy as sp

MAX_HIERARCHY_DEPTH = 10

class FormalExpression:
    """This class represents a formal expression in our NTK hierarchy system"""
    def __init__(self):
        # base sympy symbols
        self.t = sp.Symbol('t')  # time
        self.m = sp.Symbol('m')  # width
        self.H = sp.Symbol('H')  # depth
        
        self.W = sp.MatrixSymbol('W', self.m, self.m)  # weights
        self.a = sp.MatrixSymbol('a', self.m, 1)       # last layer
        # input symbols x_i
        for i in range(MAX_HIERARCHY_DEPTH):
            self.x_i = sp.Symbol(f'x_{i}') 
            
        # sigma derivatives - sigma_prime_r represents the (r+1)th derivative
        for r in range(MAX_HIERARCHY_DEPTH):
            setattr(self, f'sigma_prime_{r}', sp.Function(f'σ^({r})'))

@dataclass
class Term: # each term is a function of the entries
    def __init__(self, entry_vectors: jnp.ndarray):
        self.entry_vectors = entry_vectors # (dim_input, n_entries) in column format
        self.expr = FormalExpression()
        self.symbolic_term = self._initialize_symbolic_term()
        self.index_hierarchy = jnp.shape(self.entry_vectors)[1]

    def _initialize_symbolic_term(self) -> sp.Expr:
        """Initialize the symbolic term"""
        expr = self.expr
        # base term K^(2)
        term = (expr.a.T * expr.W / sp.sqrt(expr.m)) * expr.sigma_prime_0(expr.x_0)
        return term

    def replace_rules(self, rules: Dict[str, Any]) -> sp.Expr:
        """Applies replacement rules according to the formal rewriting system
        Args:
            rules: Dictionary containing the replacement rules to apply that are denoted a_t, W_forward, W_backward, x_layer
        Returns:
            The symbolic expression after applying the replacements
        """
        expr = self.expr
        x_beta = sp.MatrixSymbol('x_beta', self.expr.m, 1)
        
        replacements = {
            'a_t': {
                expr.a: x_beta**expr.H
            },
            'W_forward': {
                expr.W/sp.sqrt(expr.m): (
                    sp.diag(expr.sigma_prime_0(x_beta)) * 
                    (expr.W.T/sp.sqrt(expr.m)) * 
                    expr.sigma_prime_0(x_beta) * 
                    (expr.a/sp.sqrt(expr.m))
                ) * (sp.ones(1, expr.m)/sp.sqrt(expr.m)) * x_beta.T
            },
            'W_backward': {
                expr.W.T/sp.sqrt(expr.m): (
                    x_beta/sp.sqrt(expr.m) * 
                    (expr.a.T/sp.sqrt(expr.m)) * 
                    expr.sigma_prime_0(x_beta)
                )
            },
            'x_layer': {
                expr.x: sum(
                    sp.diag(
                        expr.sigma_prime_i(expr.x) * 
                        (expr.W/sp.sqrt(expr.m)) * 
                        expr.sigma_prime_1(x_beta) * 
                        (expr.W.T/sp.sqrt(expr.m)) * 
                        expr.sigma_prime_1(x_beta) * 
                        (expr.a/sp.sqrt(expr.m))
                    ) * (sp.ones(1, expr.m)/sp.sqrt(expr.m)) * 
                    (expr.x.T * x_beta)
                    for k in range(1, expr.H)
                )
            },
            'sigma_prime_i': {
                expr.sigma_prime_0(expr.x): (
                    expr.sigma_prime_1(expr.x) * 
                    sp.diag(expr.sigma_prime_0(x_beta) * 
                    (expr.W.T/sp.sqrt(expr.m)) * 
                    expr.sigma_prime_0(x_beta) * 
                    (expr.a/sp.sqrt(expr.m))) * 
                    (expr.x.T * x_beta) +
                    sum(
                        expr.sigma_prime_1(expr.x) * 
                        sp.diag((expr.W/sp.sqrt(expr.m)) * 
                        expr.sigma_prime_0(expr.x) * 
                        (expr.W/sp.sqrt(expr.m)) * 
                        expr.sigma_prime_0(x_beta) * 
                        (expr.W.T/sp.sqrt(expr.m)) * 
                        expr.sigma_prime_0(x_beta) * 
                        (expr.a/sp.sqrt(expr.m))) * 
                        (expr.x.T * x_beta)
                        for k in range(1, expr.H-1)
                    )
                )
            }
        }
        
        result = self.symbolic_term
        for rule_type, replacements_dict in rules.items():
            if rule_type in replacements:
                result = result.subs(replacements[rule_type])
            else:
                raise ValueError(f"Rule type {rule_type} not implemented")
                
        return result
        
    def evolve(self,x_beta: jnp.ndarray) -> 'Term':
        """we evolve the term with the formal system evolution law"""
        entry_vectors_new = jnp.concatenate([self.entry_vectors, x_beta], axis=1) # we concat the entry vectors and the x_beta
        new_term = Term(entry_vectors_new)
        
        beta_idx = self.index_hierarchy # new index for x_beta, because of the 1 shifting of the index
        for alpha_idx in range(beta_idx):
            # we apply each possible replacement rule because of the chain rule
            for rule in ['a_t', 'W_forward', 'W_backward', 'x_layer', 'sigma_prime_i']:
                new_term.symbolic_term += self.replace_rule(rule, beta_idx, alpha_idx)
        return new_term
    
    def evaluate(self, x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Evaluate the symbolic term with numerical values"""
        # Convert symbolic expression to numerical function
        # Implementation depends on specific requirements
        pass

class Kernel:
    def __init__(self, n_entries: int, dim_input: int, entry_vectors: jnp.ndarray):
        """The Kernel class is a container for the finite width corrections kernels
            The number of layer can be changed, and this is the most interesting part
            Same for the width"""        
        """
        Small description of what the class is doing:
        - If you have a NN, you can compute the Kernel at any point
        - You can generate NN weights randomly, and evaluate also at any point.
        - You can make the kernel evolve with the formal system evolution law.
        - You can transfer this parent kernel to another Kernel, and inherits from a son kernel.
        - You can track the time evolution during training with the truncation rules.
        - You can generate a dataset over the sphere, R^d, the Torus.
        - You can compute many statistics for the Kernel
        """
        
        """ We assume NN : R^d -> R"""

        self.n_entries = n_entries
        self.dim_input = dim_input
        self.entry_vectors = entry_vectors # (dim_input, n_entries) in column format
        self.terms = [] # list of terms in the kernel
        self.dataset = None # dataset over the sphere, R^d, the Torus.
        
        self.NN = None # the NN stored, as a jax.nn.Dense object
        self.weights = None # the weights of the NN, as a jax.numpy array
        self.feature_maps = None # the feature maps of the NN, as a jax.numpy array
        self.feature_maps_derivative = None # the derivative of the feature maps of the NN, as a jax.numpy array

    def compute_kernel(self,x: jnp.ndarray): # formula to implement
        pass
    
    def generate_NN(self):
        pass

    def evolve(self): # we apply the evolution law for each term in the kernel
        terms = []
        for term in self.terms:
            terms.append(term.evolve())
        return terms

    def get_from_son(self): # we get the kernel from a son kernel
        pass

    def compute_statistics(self):
        pass

    def generate_dataset(self):
        pass
    
    def transfer_to_parent(self):
        terms = self.evolve()
        
    def time_evolution(self):
        pass