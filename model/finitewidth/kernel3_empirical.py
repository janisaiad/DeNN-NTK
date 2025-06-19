import jax
import jax.numpy as jnp
from typing import List, Dict, Any

class Kernel3Empirical:
    """
    Computes finite width K3 kernel following the neural tangent hierarchy formula.
    """
    def __init__(self, weights: List[jnp.ndarray], 
                 sigma_derivatives: Dict[int, List[jnp.ndarray]],
                 feature_maps: Dict[int, List[jnp.ndarray]]):
        """
        Initializes the K3 kernel computation.

        Args:
            weights: List of weight matrices [W^(1), ..., W^(H), a]
            sigma_derivatives: Dict mapping layer to activation derivatives
            feature_maps: Dict mapping layer to feature maps
        """
        self.weights = weights
        self.sigma_derivatives = sigma_derivatives
        self.feature_maps = feature_maps
        self.H = len(weights) - 1  # number of hidden layers
        self.m = weights[0].shape[0]  # width
        self.n_inputs = len(feature_maps[0])
        self.G_vectors = self._compute_all_G()

    def _compute_all_G(self) -> Dict[int, List[jnp.ndarray]]:
        """
        Computes all G^(ell)_mu vectors iteratively and stores them.
        G is computed backward from layer H to 1.
        """
        G = {ell: [None] * self.n_inputs for ell in range(1, self.H + 1)}
        for mu in range(self.n_inputs):
            G[self.H][mu] = self.weights[-1] / jnp.sqrt(self.m)
            for ell in reversed(range(1, self.H)):
                W_next = self.weights[ell].T / jnp.sqrt(self.m)
                sigma_prime = self.sigma_derivatives[ell + 1][mu]
                G[ell][mu] = W_next @ sigma_prime @ G[ell + 1][mu]
        return G

    def _compute_delta_x(self, p: int, gamma: int, mu: int) -> jnp.ndarray:
        """
        Computes delta_gamma x^(p)_mu iteratively.
        """
        if p == 0:
            return jnp.zeros_like(self.feature_maps[0][mu])

        delta_x = jnp.zeros_like(self.feature_maps[0][mu])
        for j in range(1, p + 1):
            x_gamma_prev = self.feature_maps[j-1][gamma]
            x_mu_prev = self.feature_maps[j-1][mu]
            G_gamma = self.G_vectors[j][gamma]
            
            source_term = jnp.dot(x_gamma_prev, x_mu_prev) * G_gamma 
            
            propagated_term = source_term
            for k in range(j, p):
                propagated_term = self.weights[k] @ propagated_term
            
            sigma_p_prime = self.sigma_derivatives[p][mu]
            delta_x += sigma_p_prime @ propagated_term
            
        return delta_x / (jnp.sqrt(self.m) ** (p))

    def _compute_delta_G(self, ell: int, gamma: int, mu: int) -> jnp.ndarray:
        """
        Computes delta_gamma G^(ell)_mu iteratively.
        """
        result = jnp.zeros_like(self.G_vectors[ell][mu])

        for p in range(ell, self.H):
            G_p_plus_1_gamma = self.G_vectors[p+1][gamma]
            x_p_gamma = self.feature_maps[p][gamma]
            x_p_mu = self.feature_maps[p][mu]

            term = (1/self.m) * G_p_plus_1_gamma * jnp.dot(x_p_gamma, x_p_mu)
            
            for k in reversed(range(ell, p)):
                W_k_plus_1_T = self.weights[k].T / jnp.sqrt(self.m)
                sigma_k_plus_1_prime = self.sigma_derivatives[k+1][mu]
                term = W_k_plus_1_T @ sigma_k_plus_1_prime @ term
            result += term

        term_a = self.feature_maps[self.H][gamma] / jnp.sqrt(self.m)
        for k in reversed(range(ell, self.H)):
            W_k_plus_1_T = self.weights[k].T / jnp.sqrt(self.m)
            sigma_k_plus_1_prime = self.sigma_derivatives[k+1][mu]
            term_a = W_k_plus_1_T @ sigma_k_plus_1_prime @ term_a
        result += term_a

        return result

    def kernel3(self, alpha: int, beta: int, gamma: int) -> float:
        """
        Computes K^(3)(x_alpha, x_beta, x_gamma) following the NTH formula.
        """
        k3 = 0.0

        delta_x_H_alpha = self._compute_delta_x(self.H, gamma, alpha)
        delta_x_H_beta = self._compute_delta_x(self.H, gamma, beta)
        x_H_alpha = self.feature_maps[self.H][alpha]
        x_H_beta = self.feature_maps[self.H][beta]
        k3 += jnp.dot(delta_x_H_alpha, x_H_beta) + jnp.dot(x_H_alpha, delta_x_H_beta)

        for ell in range(1, self.H+1):
            delta_G_alpha = self._compute_delta_G(ell, gamma, alpha)
            delta_G_beta = self._compute_delta_G(ell, gamma, beta)
            G_alpha = self.G_vectors[ell][alpha]
            G_beta = self.G_vectors[ell][beta]
            x_alpha = self.feature_maps[ell-1][alpha]
            x_beta = self.feature_maps[ell-1][beta]
            
            k3_G = (jnp.dot(delta_G_alpha, G_beta) + jnp.dot(G_alpha, delta_G_beta)) * jnp.dot(x_alpha, x_beta)

            delta_x_alpha = self._compute_delta_x(ell-1, gamma, alpha)
            delta_x_beta = self._compute_delta_x(ell-1, gamma, beta)
            
            k3_x = jnp.dot(G_alpha, G_beta) * (jnp.dot(delta_x_alpha, x_beta) + jnp.dot(x_alpha, delta_x_beta))

            k3 += k3_G + k3_x

        return k3
