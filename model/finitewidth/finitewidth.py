import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Literal



@dataclass
class Term: # each term is a function of the entries
    def __init__(self, entry_vectors: jnp.ndarray):
        self.entry_vectors = entry_vectors # (dim_input, n_entries) in column format
        self.weights = None # the weights of the term, as a jax.numpy array

    def evolve(self,x_beta: jnp.ndarray) -> 'Term':
        """we evolve the term with the formal system evolution law"""
        entry_vectors_new = jnp.concatenate([self.entry_vectors, x_beta], axis=1) # we concat the entry vectors and the x_beta
        
        ## to fill
        return Term(self.entry_vectors_new)
    
    
    
    
    
    
    
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