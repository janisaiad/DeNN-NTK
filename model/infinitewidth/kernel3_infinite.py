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
        """Calcule une entrée de la matrice du noyau K3."""
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        norm_x3 = np.linalg.norm(x3)

        if norm_x1 == 0 or norm_x2 == 0 or norm_x3 == 0:
            return 0.0

        rho12 = np.clip(np.dot(x1, x2) / (norm_x1 * norm_x2), -1.0, 1.0)
        rho13 = np.clip(np.dot(x1, x3) / (norm_x1 * norm_x3), -1.0, 1.0)
        rho23 = np.clip(np.dot(x2, x3) / (norm_x2 * norm_x3), -1.0, 1.0)

        rhos12, rhos13, rhos23 = [rho12], [rho13], [rho23]
        for _ in range(1, self.l):
            rhos12.append(self._varrho(rhos12[-1]))
            rhos13.append(self._varrho(rhos13[-1]))
            rhos23.append(self._varrho(rhos23[-1]))

        # Dérivées premières
        rho_primes12 = [self._varrho_prime(rho) for rho in rhos12]
        rho_primes13 = [self._varrho_prime(rho) for rho in rhos13]
        rho_primes23 = [self._varrho_prime(rho) for rho in rhos23]

        # Dérivées secondes
        rho_double_primes12 = [self._varrho_double_prime(rho) for rho in rhos12]
        
        k3_sum = 0
        for k in range(1, self.l + 1):
            prod_primes_12 = np.prod(rho_primes12[k-1:self.l-1])
            prod_primes_13 = np.prod(rho_primes13[k-1:self.l-1])
            prod_primes_23 = np.prod(rho_primes23[k-1:self.l-1])

            # Terme de base de K3, dérivé de K2
            # La formule exacte est complexe, ceci est une approximation structurelle
            # basée sur la différentiation de la formule K2
            
            # Contribution de la différentiation de rhos[k-1]
            term1 = rho_primes12[k-1] * rhos13[k-1] * prod_primes_12 * prod_primes_13
            
            # Contribution de la différentiation de prod(rho_primes)
            sum_over_j = 0
            for j in range(k - 1, self.l - 1):
                # d/d_theta(rho_primes[j])
                term_j = rho_double_primes12[j] * rho_primes13[j]
                
                prod_without_j = np.prod([p for idx, p in enumerate(rho_primes12[k-1:self.l-1]) if idx != j-(k-1)])
                sum_over_j += term_j * prod_without_j
            
            term2 = rhos12[k-1] * rhos23[k-1] * sum_over_j * prod_primes_23
            
            k3_sum += term1 + term2

        # La symétrie complète devrait être assurée en permutant x1, x2, x3
        # Pour simplifier, nous ne montrons que la différentiation par rapport aux paramètres
        # qui affectent le chemin x1-x2 et contracté avec le gradient de f(x3)
        return norm_x1 * norm_x2 * norm_x3 * k3_sum

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
