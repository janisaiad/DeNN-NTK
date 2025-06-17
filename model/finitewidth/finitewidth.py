import jax.numpy as jnp

class Kernel:
    def __init__(self, n_entries: int, dim_input: int, entry_vectors: np.ndarray):
        "The Kernel class is a container for the finite width corrections kernels"
        
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
        

    def compute_kernel(self): # formula to implement
        pass

    def compute_kernel_from_weights(self):
        pass

    def evolve(self): # we apply the evolution law for each term in the kernel
        pass

    def evaluate(self):
        pass

