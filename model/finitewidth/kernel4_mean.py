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

    def _compute_delta_x(self, p: int, gamma: int, mu: int) -> jnp.ndarray:
        """
        Computes delta_gamma x^(p)_mu recursively.
        """
        if p == 0:
            return jnp.zeros_like(self.feature_maps[0][mu])  # base case: delta_gamma x^(0)_mu = 0

        sigma_p_prime = self.sigma_derivatives[p][mu]  # get activation derivative sigma'_p(x_mu)

        delta_x_prev = self._compute_delta_x(p-1, gamma, mu)  
        term1 = self.weights[p-1] @ delta_x_prev  # compute W^(p) (delta_gamma x^(p-1)_mu)

        x_gamma = self.feature_maps[p-1][gamma]
        x_mu = self.feature_maps[p-1][mu]
        G_gamma = self._compute_G(p, gamma)
        term2 = jnp.dot(x_gamma, x_mu) * G_gamma  # compute <x^(p-1)_gamma, x^(p-1)_mu> G^(p)_gamma

        return (1/jnp.sqrt(self.m)) * sigma_p_prime @ (term1 + term2)

    def _compute_G(self, ell: int, mu: int) -> jnp.ndarray:
        """
        Computes G^(ell)_mu.
        """
        if ell == self.H:
            return self.weights[-1] / jnp.sqrt(self.m)

        G_next = self._compute_G(ell+1, mu)  # recursively compute G through layers
        W_next = self.weights[ell].T / jnp.sqrt(self.m)
        sigma_prime = self.sigma_derivatives[ell+1][mu]
        
        return W_next @ sigma_prime @ G_next

    def _compute_delta_G(self, ell: int, gamma: int, mu: int) -> jnp.ndarray:
        """
        Computes delta_gamma G^(ell)_mu.
        """
        result = jnp.zeros_like(self._compute_G(ell, mu))

        for p in range(ell, self.H+1):  # sum over layers p from ell to H
            if p < self.H:  # term from W^(p+1) replacement
                G_gamma = self._compute_G(p+1, gamma)
                x_gamma = self.feature_maps[p][gamma]
                x_mu = self.feature_maps[p][mu]
                term = jnp.outer(G_gamma, x_gamma) @ x_mu / self.m
                
                for k in range(ell, p):  # propagate through remaining layers
                    term = (self.weights[k].T / jnp.sqrt(self.m)) @ self.sigma_derivatives[k+1][mu] @ term
                result += term

        if ell <= self.H:  # term from a_t replacement (x^(H)_gamma)
            term = self.feature_maps[self.H][gamma] / jnp.sqrt(self.m)
            for k in range(ell, self.H):
                term = (self.weights[k].T / jnp.sqrt(self.m)) @ self.sigma_derivatives[k+1][mu] @ term
            result += term

        return result

    def kernel3(self, alpha: int, beta: int, gamma: int) -> float:
        """
        Computes K^(3)(x_alpha, x_beta, x_gamma) following the NTH formula.
        """
        k3 = 0.0

        delta_x_H_alpha = self._compute_delta_x(self.H, gamma, alpha)  # k^(3,out) term
        delta_x_H_beta = self._compute_delta_x(self.H, gamma, beta)
        x_H_alpha = self.feature_maps[self.H][alpha]
        x_H_beta = self.feature_maps[self.H][beta]
        k3 += jnp.dot(delta_x_H_alpha, x_H_beta) + jnp.dot(x_H_alpha, delta_x_H_beta)

        for ell in range(1, self.H+1):  # sum over layers for k^(3,g) and k^(3,x) terms
            delta_G_alpha = self._compute_delta_G(ell, gamma, alpha)  # k^(3,g) term
            delta_G_beta = self._compute_delta_G(ell, gamma, beta)
            G_alpha = self._compute_G(ell, alpha)
            G_beta = self._compute_G(ell, beta)
            x_alpha = self.feature_maps[ell-1][alpha]
            x_beta = self.feature_maps[ell-1][beta]
            
            k3_G = (jnp.dot(delta_G_alpha, G_beta) + jnp.dot(G_alpha, delta_G_beta)) * jnp.dot(x_alpha, x_beta)

            delta_x_alpha = self._compute_delta_x(ell-1, gamma, alpha)  # k^(3,x) term
            delta_x_beta = self._compute_delta_x(ell-1, gamma, beta)
            
            k3_x = jnp.dot(G_alpha, G_beta) * (jnp.dot(delta_x_alpha, x_beta) + jnp.dot(x_alpha, delta_x_beta))

            k3 += k3_G + k3_x

        return k3
