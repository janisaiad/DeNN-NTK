import numpy as np

class Kernel3Infinite:
    def __init__(self, n_layers: int, n_outputs: int, a: float = 1.0, b: float = 1):
        """
        we initialize the infinite width model.

        Args:
            n_layers (int): number of layers, l.
            n_outputs (int): output dimension, m_l.
            a (float): parameter 'a' of the (a,b)-ReLU activation function.
            b (float): parameter 'b' of the (a,b)-ReLU activation function.
        """
        self.l = n_layers
        self.ml = n_outputs
        self.a = a
        self.b = b
        
        if self.a**2 + self.b**2 == 0:
            raise ValueError("a^2 + b^2 ne peut pas être nul.")
            
        self.delta_phi = self.b**2 / (self.a**2 + self.b**2)
        # sigma est défini par l'initialisation EOC (Edge Of Chaos)
        self.sigma = (self.a**2 + self.b**2)**-0.5

    def _varrho(self, rho: np.ndarray) -> np.ndarray:
        """cosine map"""
        rho = np.clip(rho, -1.0, 1.0)
        return rho + self.delta_phi * (2 / np.pi) * (np.sqrt(1 - rho**2) - rho * np.arccos(rho))

    def _varrho_prime(self, rho: np.ndarray) -> np.ndarray:
        """derivative of the cosine map"""
        rho = np.clip(rho, -1.0, 1.0)
        return 1 - self.delta_phi * (2 / np.pi) * np.arccos(rho)

    def _varrho_double_prime(self, rho: np.ndarray) -> np.ndarray:
        """second derivative of the cosine map"""
        # we clip for stability
        rho_clipped = np.clip(rho, -1.0 + 1e-5, 1.0 - 1e-5) #  e-5 coherent with our dataset size
        return self.delta_phi * (2 / np.pi) / np.sqrt(1 - rho_clipped**2)

    def _kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """one entry of the NTK matrix"""
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 0.0

        rho1 = np.dot(x1, x2) / (norm_x1 * norm_x2)
        rho1 = np.clip(rho1, -1.0, 1.0)
        
        rhos = [rho1]
        for _ in range(1, self.l):
            rhos.append(self._varrho(rhos[-1])) # because we do it recursively
            
        rho_primes = [self._varrho_prime(rho) for rho in rhos]

        k_sum = 0
        for k in range(1, self.l + 1):
            # rho_k is rhos[k-1]
            # product of rho_primes from k'=k to l-1, which are the indices k-1 to l-2 of rho_primes
            prod = np.prod(rho_primes[k-1:self.l-1])
            term = rhos[k-1] * prod
            k_sum += term
            
        return norm_x1 * norm_x2 * k_sum

    def _kernel_3_entry(self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
        """
        computes one entry of the k3 kernel matrix.
        this follows the recursive structure from the neural tangent hierarchy.
        """
        # norms of inputs
        c1 = np.dot(x1, x1)
        c2 = np.dot(x2, x2)
        c3 = np.dot(x3, x3)
        
        # initial covariances (layer 0)
        c12 = np.dot(x1, x2)
        c13 = np.dot(x1, x3)
        c23 = np.dot(x2, x3)

        # initial kernels (layer 0 is just input, so kernel is 0)
        k12, k13, k23, k33 = 0.0, 0.0, 0.0, 0.0
        k3_123 = 0.0
        
        # iterate through layers
        for _ in range(self.l):
            # cosines from previous layer's covariances
            rho12 = c12 / np.sqrt(c1 * c2)
            rho13 = c13 / np.sqrt(c1 * c3)
            rho23 = c23 / np.sqrt(c2 * c3)
            
            # map derivatives
            dp12 = self._varrho_prime(rho12)
            dp13 = self._varrho_prime(rho13)
            dp23 = self._varrho_prime(rho23)
            d2p12 = self._varrho_double_prime(rho12)
            
            # update k3
            # this is a direct implementation of the recursive formula for k3
            # derived from differentiating the k2 recursion
            
            # derivative of c12 w.r.t theta, contracted with grad f(x3)
            dc12_d3 = 0.5 * (k13 * dp13 / c1 * c12 + k23 * dp23 / c2 * c12 - k33 * dp13 * dp23 / c3 * c12 + k3_123)
            
            # update k3_123 for the next layer
            k3_123 = k3_123 * dp12 * dp13 * dp23 + \
                     k12 * d2p12 * dc12_d3 + \
                     c12 * (d2p12 * dp13 * dp23) * k3_123

            # update covariances for next layer
            c1_new = c1 * self._varrho(1.0)
            c2_new = c2 * self._varrho(1.0)
            c3_new = c3 * self._varrho(1.0)
            c12_new = np.sqrt(c1 * c2) * self._varrho(rho12)
            c13_new = np.sqrt(c1 * c3) * self._varrho(rho13)
            c23_new = np.sqrt(c2 * c3) * self._varrho(rho23)
            
            c1, c2, c3 = c1_new, c2_new, c3_new
            c12, c13, c23 = c12_new, c13_new, c23_new
            
            # update k2 for next layer
            k12_new = k12 * dp12 + c12
            k13_new = k13 * dp13 + c13
            k23_new = k23 * dp23 + c23
            k33_new = k33 * 1.0 + c3 # dp for rho=1 is 1
            
            k12, k13, k23, k33 = k12_new, k13_new, k23_new, k33_new

        return k3_123

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        compute the NTK matrix for a given set of data

        Args:
            X (np.ndarray): data matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: the NTK matrix of shape (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self._kernel_entry(X[i], X[j])
                K[j, i] = K[i, j]
        
        # the paper defines K as [1/n * K(xi, xj)].
        # I return K(xi,xj) here. The normalization can be done outside.
        return K
