import numpy as np

class InfiniteWidth:
    def __init__(self, n_layers: int, n_outputs: int, a: float = 1.0, b: float = 1.0):
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
        """Calcule la carte cosinus."""
        rho = np.clip(rho, -1.0, 1.0)
        return rho + self.delta_phi * (2 / np.pi) * (np.sqrt(1 - rho**2) - rho * np.arccos(rho))

    def _varrho_prime(self, rho: np.ndarray) -> np.ndarray:
        """Calcule la dérivée de la carte cosinus."""
        rho = np.clip(rho, -1.0, 1.0)
        return 1 - self.delta_phi * (2 / np.pi) * np.arccos(rho)

    def _kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcule une seule entrée K(x1, x2) de la matrice NTK.

        Args:
            x1 (np.ndarray): Premier vecteur d'entrée.
            x2 (np.ndarray): Second vecteur d'entrée.

        Returns:
            float: La valeur de l'entrée du noyau.
        """
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 0.0

        rho1 = np.dot(x1, x2) / (norm_x1 * norm_x2)
        rho1 = np.clip(rho1, -1.0, 1.0)
        
        rhos = [rho1]
        for _ in range(1, self.l):
            rhos.append(self._varrho(rhos[-1]))
            
        rho_primes = [self._varrho_prime(rho) for rho in rhos]

        k_sum = 0
        for k in range(1, self.l + 1):
            # rho_k est rhos[k-1]
            # Produit de rho_primes de k'=k à l-1, qui sont les indices k-1 à l-2 de rho_primes
            prod = np.prod(rho_primes[k-1:self.l-1])
            term = rhos[k-1] * prod
            k_sum += term
            
        return norm_x1 * norm_x2 * k_sum

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice NTK pour un ensemble de données donné.

        Args:
            X (np.ndarray): Matrice de données de forme (n_samples, n_features).

        Returns:
            np.ndarray: La matrice NTK de forme (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self._kernel_entry(X[i], X[j])
                K[j, i] = K[i, j]
        
        # Le papier définit K comme [1/n * K(xi, xj)].
        # Je retourne K(xi,xj) ici. La normalisation peut être faite à l'extérieur.
        return K

    def fit(self):
        """Non applicable pour le NTK à largeur infinie car le noyau est fixe."""
        pass

    def predict(self):
        """La prédiction se ferait en utilisant la régression par noyau, en dehors de cette classe."""
        pass

    def evaluate(self):
        """L'évaluation se ferait en utilisant la régression par noyau, en dehors de cette classe."""
        pass

