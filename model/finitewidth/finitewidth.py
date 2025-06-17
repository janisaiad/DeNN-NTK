import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Literal
import sympy as sp

MAX_HIERARCHY_DEPTH = 10
MAX_LAYERS = 100 # not well optimized yet but it works

class FormalExpression:
    """This class represents a formal expression in our NTK hierarchy system"""
    def __init__(self):
        # base sympy symbols
        self.t = sp.Symbol('t')  # time
        self.m = sp.Symbol('m')  # width
        
        self.W_symbols = {}
        for layer in range(MAX_LAYERS+1):
            self.W_symbols[str(layer)] = sp.MatrixSymbol('W', self.m, self.m)  # weights
        
        self.a = sp.MatrixSymbol('a', self.m, 1)       # last layer
        # input symbols x_i
        
        self.x_symbols = {}
        for i in range(MAX_HIERARCHY_DEPTH):
            for layer in range(MAX_LAYERS+1): # +1 because of the last layer
                self.x_symbols[str(i)+'_'+str(layer)] = sp.Symbol(f'x_{i}^{layer}') 
            
        # sigma and its derivatives - sigma_prime_r represents the (r+1)th derivative
        self.sigma_prime_symbols = {}
        for r in range(MAX_HIERARCHY_DEPTH):
            for layer in range(MAX_LAYERS+1):
                self.sigma_prime_symbols[str(r)+'_'+str(layer)] = sp.Function(f'σ_{layer}^({r})')



@dataclass
class Term: # each term is a function of the entries
    def __init__(self, entry_vectors: jnp.ndarray, H: int):
        self.entry_vectors = entry_vectors # (dim_input, n_entries) in column format
        self.expr = FormalExpression()
        self.index_hierarchy = jnp.shape(self.entry_vectors)[1]
        self.H = H
        self.symbolic_term = self._initialize_symbolic_term()
        
    def _initialize_symbolic_term(self) -> sp.Expr:
        """Initialize the symbolic term"""
        expr = self.expr
        # base term for f(x, theta)
        term = expr.a.T*expr.x_symbols[str(0)+'_'+str(self.H)]
        return term



    def replace_rules(self, rules_to_apply: Dict[str, bool]) -> sp.Expr:
        """
        Applies replacement rules simultaneously according to the formal rewriting system.
        This is done in a single pass to avoid sequential substitution errors.

        Args:
            rules_to_apply: Dictionary indicating which types of rules to apply (e.g., {'a_t': True, 'W_forward': True}).

        Returns:
            The symbolic expression after applying all specified replacements at once.
        """
        expr = self.expr
        m = self.expr.m
        x_beta = {}  # we extract the dictionary for the last layer

        # not well optimized yet but it works
        for layer in range(MAX_LAYERS + 1):  # if H = 2, then layer = 0, 1, 2 which is coherent, the last is the last layer
            x_beta[str(layer)] = sp.MatrixSymbol(f'x_beta^{{layer}}', m, 1)

        # Helper to build the forward propagation chain from the paper
        def _build_forward_chain(start_layer: int) -> sp.Expr:
            chain = sp.Identity(m)
            # Chain of W.T and sigma'
            for k in range(start_layer, self.H):
                s_prime_k_func = expr.sigma_prime_symbols[f'0_{k}']
                s_prime_k_vec = x_beta[str(k)].applyfunc(s_prime_k_func)
                W_kp1_T = expr.W_symbols[str(k + 1)].T
                chain = chain * sp.diag(s_prime_k_vec) * (W_kp1_T / sp.sqrt(m))
            # Last layer application
            s_prime_H_func = expr.sigma_prime_symbols[f'0_{self.H}']
            s_prime_H_vec = x_beta[str(self.H)].applyfunc(s_prime_H_func)
            chain = chain * sp.diag(s_prime_H_vec) * (expr.a / sp.sqrt(m))
            return chain

        all_replacements = {
            'a_t': {
                expr.a: x_beta[str(self.H - 1)]  # not the final value !
            },
            'W_forward': {
                expr.W_symbols[str(layer)] / sp.sqrt(m): (
                    sp.diag(_build_forward_chain(layer)) * 
                    (sp.ones(m, 1) / sp.sqrt(m)) * 
                    x_beta[str(layer - 1)].T
                )
                for layer in range(1, self.H)
            },
            'W_backward': {
                expr.W_symbols[str(layer)].T / sp.sqrt(m): (
                    x_beta[str(layer - 1)] / sp.sqrt(m) *
                    (expr.a.T / sp.sqrt(m)) *
                    _build_forward_chain(layer).T
                )
                for layer in range(1, self.H + 1)
            },
            'x_layer': {
                expr.x_symbols[str(i) + '_' + str(layer)]: sum(
                    sp.diag(
                        expr.sigma_prime_symbols[str(0) + '_' + str(layer)](expr.x_symbols[str(i) + '_' + str(layer)]) *
                        (expr.W_symbols[str(layer + 1)] / sp.sqrt(m)) *
                        expr.sigma_prime_symbols[str(0) + '_' + str(k)](expr.x_symbols[str(i) + '_' + str(k)]) *
                        expr.sigma_prime_symbols[str(0) + '_' + str(k)](x_beta[str(k)]) *
                        (expr.W_symbols[str(k + 1)].T / sp.sqrt(m)) *
                        expr.sigma_prime_symbols[str(0) + '_' + str(self.H)](x_beta[str(self.H)]) *
                        (expr.a / sp.sqrt(m))
                    ) * (sp.ones(1, m) / sp.sqrt(m)) *
                    (expr.x_symbols[str(i) + '_' + str(k - 1)].T * x_beta[str(k - 1)])
                    for k in range(1, layer + 1)
                )
                for i in range(self.index_hierarchy)
                for layer in range(1, self.H)
            },
            'sigma_prime_i': {
                expr.sigma_prime_symbols[str(r) + '_' + str(layer)](expr.x_symbols[str(i) + '_' + str(layer)]): (
                    expr.sigma_prime_symbols[str(r + 1) + '_' + str(layer)](expr.x_symbols[str(i) + '_' + str(layer)]) *
                    sp.diag(
                        expr.sigma_prime_symbols[str(0) + '_' + str(layer)](x_beta[str(layer)]) *
                        (expr.W_symbols[str(layer + 1)].T / sp.sqrt(m)) *
                        expr.sigma_prime_symbols[str(0) + '_' + str(layer + 1)](x_beta[str(layer + 1)]) *
                        (expr.a / sp.sqrt(m))
                    ) * (expr.x_symbols[str(i) + '_' + str(layer - 1)].T * x_beta[str(layer - 1)]) +
                    sum(
                        expr.sigma_prime_symbols[str(r + 1) + '_' + str(layer)](expr.x_symbols[str(i) + '_' + str(layer)]) *
                        sp.diag(
                            (expr.W_symbols[str(layer)] / sp.sqrt(m)) *
                            expr.sigma_prime_symbols[str(0) + '_' + str(k)](expr.x_symbols[str(i) + '_' + str(k)]) *
                            expr.sigma_prime_symbols[str(0) + '_' + str(k)](x_beta[str(k)]) *
                            (expr.W_symbols[str(k + 1)].T / sp.sqrt(m)) *
                            expr.sigma_prime_symbols[str(0) + '_' + str(self.H)](x_beta[str(self.H)]) *
                            (expr.a / sp.sqrt(m))
                        ) * (expr.x_symbols[str(i) + '_' + str(k - 1)].T * x_beta[str(k - 1)])
                        for k in range(1, layer)
                    )
                )
                for i in range(self.index_hierarchy)
                for r in range(MAX_HIERARCHY_DEPTH - 1)
                for layer in range(1, self.H)
            }
        }

        # 1. On fusionne tous les dictionnaires de règles demandées en un seul.
        final_replacements = {}
        for rule_type, should_apply in rules_to_apply.items():
            if should_apply and rule_type in all_replacements:
                final_replacements.update(all_replacements[rule_type])

        # 2. On appelle .subs() une seule et unique fois avec le dictionnaire complet.
        # Sympy effectue alors un remplacement simultané.
        return self.symbolic_term.subs(final_replacements, simultaneous=True)

    def evolve(self, x_beta: jnp.ndarray) -> 'Term':
        """we evolve the term with the formal system evolution law
        Args:
            x_beta: The new input vector to add to the term's entry vectors
        Returns:
            A new Term with all rewriting rules applied simultaneously
        """
        entry_vectors_new = jnp.concatenate([self.entry_vectors, x_beta], axis=1)
        new_term = Term(entry_vectors_new, self.H)
        
        all_rules = {
            'a_t': True,
            'W_forward': True,
            'W_backward': True,
            'x_layer': True,
            'sigma_prime_i': True
        }
        
        new_term.symbolic_term = self.replace_rules(all_rules)
        
        return new_term
    
    def evaluate(self, x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Evaluate the symbolic term with numerical values"""
        # Convert symbolic expression to numerical function
        # Implementation depends on specific requirements
        pass







class Kernel:
    def __init__(self, n_entries: int, dim_input: int, entry_vectors: jnp.ndarray, H: int):
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
        self.H = H
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