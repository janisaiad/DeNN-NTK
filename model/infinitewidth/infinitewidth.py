import jax.numpy as jnp

try:
    import os
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # we force GPU usage globally

    import jax
    jax.config.update('jax_platform_name', 'gpu')  # we configure JAX for GPU
except:
    print("No GPU found, using CPU")
    pass

class NtkInfiniteWidth:
    def __init__(self, n_layers: int, n_outputs: int, a: float = 1.0, b: float = 1):
        """
        Initialize the infinite width model.

        Args:
            n_layers (int): Number of layers, l.
            n_outputs (int): Output dimension, m_l.
            a (float): Parameter 'a' of the (a,b)-ReLU activation function.
            b (float): Parameter 'b' of the (a,b)-ReLU activation function.
        """
        self.l = n_layers
        self.ml = n_outputs
        self.a = a
        self.b = b
        
        if self.a**2 + self.b**2 == 0:
            raise ValueError("a^2 + b^2 cannot be zero.")
            
        self.delta_phi = self.b**2 / (self.a**2 + self.b**2)
        # sigma is defined by EOC (Edge Of Chaos) initialization
        self.sigma = (self.a**2 + self.b**2)**-0.5

    def _varrho(self, rho: jnp.ndarray) -> jnp.ndarray:
        """cosine map"""
        rho = jnp.clip(rho, -1.0, 1.0)
        return rho + self.delta_phi * (2 / jnp.pi) * (jnp.sqrt(1 - rho**2) - rho * jnp.arccos(rho))

    def _varrho_prime(self, rho: jnp.ndarray) -> jnp.ndarray:
        """derivative of the cosine map"""
        rho = jnp.clip(rho, -1.0, 1.0)
        return 1 - self.delta_phi * (2 / jnp.pi) * jnp.arccos(rho)

    def _kernel_entry(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """one entry of the NTK matrix"""
        norm_x1 = jnp.linalg.norm(x1)
        norm_x2 = jnp.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 0.0

        rho1 = jnp.dot(x1, x2) / (norm_x1 * norm_x2)
        rho1 = jnp.clip(rho1, -1.0, 1.0)
        
        rhos = [rho1]
        for _ in range(1, self.l):
            rhos.append(self._varrho(rhos[-1])) # because we do it recursively
            
        rho_primes = [self._varrho_prime(rho) for rho in rhos]

        k_sum = 0
        for k in range(1, self.l + 1):
            # rho_k is rhos[k-1]
            # product of rho_primes from k'=k to l-1, which are the indices k-1 to l-2 of rho_primes
            if k-1 < self.l-1:  # we check if slice is non-empty
                rho_primes_slice = jnp.array(rho_primes[k-1:self.l-1])  # we convert list slice to array
                prod = jnp.prod(rho_primes_slice)  # we compute product
            else:
                prod = 1.0  # we set default value for empty slice
            term = rhos[k-1] * prod
            k_sum += term
            
        return norm_x1 * norm_x2 * k_sum

    def kernel_matrix(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        compute the NTK matrix for a given set of data

        Args:
            X (np.ndarray): data matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: the NTK matrix of shape (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        K = jnp.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                entry_value = self._kernel_entry(X[i], X[j])  # we compute kernel entry
                K = K.at[i, j].set(entry_value)  # we set upper triangular part
                K = K.at[j, i].set(entry_value)  # we set lower triangular part for symmetry
        
        # the paper defines K as [1/n * K(xi, xj)].
        # I return K(xi,xj) here. The normalization can be done outside.
        return K
