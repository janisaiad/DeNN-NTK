import numpy as np

class Kernel3Infinite:
    def __init__(self, n_layers: int, n_outputs: int, a: float = 1.0, b: float = 1):
        """
        Initialise le modèle de largeur infinie.

        Args:
            n_layers (int): Nombre de couches, l.
            n_outputs (int): Dimension de la sortie, m_l.
            a (float): Paramètre 'a' de la fonction d'activation (a,b)-ReLU.
            b (float): Paramètre 'b' de la fonction d'activation (a,b)-ReLU.
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
