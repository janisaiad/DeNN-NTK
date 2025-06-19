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
        self.H = len(weights) - 1  # we get the number of hidden layers
        self.m = weights[0].shape[0]  # we get the width
        self.n_inputs = len(feature_maps[0]) # we get the number of inputs

    def kernel3(self, alpha: int, beta: int, gamma: int) -> float:
        """
        Computes K^(3)(x_alpha, x_beta, x_gamma) using the fully expanded non-recursive formula.
        """
        k3 = 0.0 # we initialize the k3 value

        # we pre-compute G vectors for all layers and relevant inputs to avoid re-computation
        G = {mu: {} for mu in [alpha, beta, gamma]}
        for mu in [alpha, beta, gamma]:
            G[mu][self.H] = self.weights[-1] / jnp.sqrt(self.m) # we compute G for the last layer H
            for ell in reversed(range(1, self.H)): # we compute G for other layers by iterating backwards
                G_next = G[mu][ell+1]
                W_next = self.weights[ell].T
                sigma_prime = self.sigma_derivatives[ell+1][mu]
                G[mu][ell] = (W_next @ (sigma_prime * G_next)) / jnp.sqrt(self.m)

        # we pre-compute K^(3,out) term
        for j in range(1, self.H + 1):
            # we compute term for x_alpha
            J_alpha = jnp.identity(self.m) # we compute the forward propagator J^(H->j)_alpha
            for k in range(H, j, -1):
                J_alpha = J_alpha @ (self.sigma_derivatives[k][alpha] * self.weights[k-1]) / jnp.sqrt(self.m)
            
            x_jm1_gamma = self.feature_maps[j-1][gamma] # we compute the source term T^(j)_gamma,alpha
            x_jm1_alpha = self.feature_maps[j-1][alpha]
            T_alpha = self.sigma_derivatives[j][alpha] * (jnp.dot(x_jm1_gamma, x_jm1_alpha) * G[gamma][j]) / jnp.sqrt(self.m)
            
            delta_x_H_alpha = J_alpha @ T_alpha
            x_H_beta = self.feature_maps[self.H][beta]
            k3 += jnp.dot(delta_x_H_alpha, x_H_beta)

            # we compute term for x_beta
            J_beta = jnp.identity(self.m) # we compute the forward propagator J^(H->j)_beta
            for k in range(self.H, j, -1):
                J_beta = J_beta @ (self.sigma_derivatives[k][beta] * self.weights[k-1]) / jnp.sqrt(self.m)
            
            x_jm1_beta = self.feature_maps[j-1][beta] # we compute the source term T^(j)_gamma,beta
            T_beta = self.sigma_derivatives[j][beta] * (jnp.dot(x_jm1_gamma, x_jm1_beta) * G[gamma][j]) / jnp.sqrt(self.m)

            delta_x_H_beta = J_beta @ T_beta
            x_H_alpha = self.feature_maps[self.H][alpha]
            k3 += jnp.dot(x_H_alpha, delta_x_H_beta)

        # we pre-compute K^(3,G) and K^(3,x) terms
        for ell in range(1, self.H + 1):
            # we compute K^(3,G) part
            k3_G_term = 0.0
            for p in range(ell, self.H): # we sum over p from ell to H-1
                # we compute term for G_alpha
                B_alpha = jnp.identity(self.m) # we compute the backward propagator B^(ell->p)_alpha
                for k in range(p, ell - 1, -1):
                    B_alpha = B_alpha @ (self.weights[k].T * self.sigma_derivatives[k+1][alpha]) / jnp.sqrt(self.m)
                
                x_p_gamma = self.feature_maps[p][gamma] # we compute the source term U^(p)_gamma,alpha
                x_p_alpha = self.feature_maps[p][alpha]
                U_alpha = (G[gamma][p+1] * jnp.dot(x_p_gamma, x_p_alpha)) / self.m

                delta_G_p_alpha = B_alpha @ U_alpha
                k3_G_term += jnp.dot(delta_G_p_alpha, G[beta][ell])

                # we compute term for G_beta
                B_beta = jnp.identity(self.m) # we compute the backward propagator B^(ell->p)_beta
                for k in range(p, ell - 1, -1):
                    B_beta = B_beta @ (self.weights[k].T * self.sigma_derivatives[k+1][beta]) / jnp.sqrt(self.m)

                x_p_beta = self.feature_maps[p][beta] # we compute the source term U^(p)_gamma,beta
                U_beta = (G[gamma][p+1] * jnp.dot(x_p_gamma, x_p_beta)) / self.m
                
                delta_G_p_beta = B_beta @ U_beta
                k3_G_term += jnp.dot(G[alpha][ell], delta_G_p_beta)

            # we add the term from differentiating the output weights 'a'
            B_H_alpha = jnp.identity(self.m)
            for k in range(self.H - 1, ell - 1, -1):
                B_H_alpha = B_H_alpha @ (self.weights[k].T * self.sigma_derivatives[k+1][alpha]) / jnp.sqrt(self.m)
            
            x_H_gamma = self.feature_maps[self.H][gamma]
            delta_G_a_alpha = B_H_alpha @ (x_H_gamma / jnp.sqrt(self.m))
            k3_G_term += jnp.dot(delta_G_a_alpha, G[beta][ell])

            B_H_beta = jnp.identity(self.m)
            for k in range(self.H - 1, ell - 1, -1):
                B_H_beta = B_H_beta @ (self.weights[k].T * self.sigma_derivatives[k+1][beta]) / jnp.sqrt(self.m)
            
            delta_G_a_beta = B_H_beta @ (x_H_gamma / jnp.sqrt(self.m))
            k3_G_term += jnp.dot(G[alpha][ell], delta_G_a_beta)
            
            x_ellm1_alpha = self.feature_maps[ell-1][alpha]
            x_ellm1_beta = self.feature_maps[ell-1][beta]
            k3 += k3_G_term * jnp.dot(x_ellm1_alpha, x_ellm1_beta)

            # we compute K^(3,x) part
            k3_x_term = 0.0
            for j in range(1, ell):
                # we compute term for x_alpha
                J_alpha = jnp.identity(self.m) # we compute the forward propagator J^(ell-1 -> j)_alpha
                for k in range(ell - 1, j, -1):
                    J_alpha = J_alpha @ (self.sigma_derivatives[k][alpha] * self.weights[k-1]) / jnp.sqrt(self.m)

                x_jm1_gamma = self.feature_maps[j-1][gamma] # we compute the source term T^(j)_gamma,alpha
                x_jm1_alpha = self.feature_maps[j-1][alpha]
                T_alpha = self.sigma_derivatives[j][alpha] * (jnp.dot(x_jm1_gamma, x_jm1_alpha) * G[gamma][j]) / jnp.sqrt(self.m)
                
                delta_x_alpha = J_alpha @ T_alpha
                x_ellm1_beta = self.feature_maps[ell-1][beta]
                k3_x_term += jnp.dot(delta_x_alpha, x_ellm1_beta)
                
                # we compute term for x_beta
                J_beta = jnp.identity(self.m) # we compute the forward propagator J^(ell-1 -> j)_beta
                for k in range(ell - 1, j, -1):
                    J_beta = J_beta @ (self.sigma_derivatives[k][beta] * self.weights[k-1]) / jnp.sqrt(self.m)

                x_jm1_beta = self.feature_maps[j-1][beta] # we compute the source term T^(j)_gamma,beta
                T_beta = self.sigma_derivatives[j][beta] * (jnp.dot(x_jm1_gamma, x_jm1_beta) * G[gamma][j]) / jnp.sqrt(self.m)

                delta_x_beta = J_beta @ T_beta
                x_ellm1_alpha = self.feature_maps[ell-1][alpha]
                k3_x_term += jnp.dot(x_ellm1_alpha, delta_x_beta)
            
            k3 += k3_x_term * jnp.dot(G[alpha][ell], G[beta][ell])

        return k3
