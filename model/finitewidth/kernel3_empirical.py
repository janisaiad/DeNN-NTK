import jax
import jax.numpy as jnp
from typing import List, Dict, Any

## This will be the empirical computations for K3
## we will use a special parabola arch to compute the kernel to maximize the concentration I guess


class Kernel3Empirical:
    """
    Computes finite width kernels K_2 (NTK) and K_3 for a fully-connected neural network.
    """
    def __init__(self, weights: List[jnp.ndarray], 
                 sigma_derivatives: Dict[int, List[jnp.ndarray]],
                 feature_maps: Dict[int, List[jnp.ndarray]]):
        """
        Initializes the kernel computation module.

        Args:
            weights: List of weight matrices [W^(1), ..., W^(H), a].
            sigma_derivatives: Dictionary mapping layer index to activation derivatives for each input.
            feature_maps: Dictionary mapping layer index to feature maps for each input.
        """
        self.weights = weights
        self.sigma_derivatives = sigma_derivatives
        self.feature_maps = feature_maps
        self.H = len(weights) - 1
        self.m = weights[0].shape[0]
        self.n_inputs = len(feature_maps[0])
        
        # Memoization caches
        self._memo_p = {}
        self._memo_p_repl_a = {}
        self._memo_x_dot = {}
        self._memo_p_repl_W = {}

    def _compute_p(self, ell: int, alpha_idx: int) -> jnp.ndarray:
        """Helper to compute the vector p_ell for a given input alpha_idx, using memoization."""
        if (ell, alpha_idx) in self._memo_p:
            return self._memo_p[(ell, alpha_idx)]

        if ell == self.H:
            # p_H = a / sqrt(m)
            res = self.weights[self.H] / jnp.sqrt(self.m)
        else:
            # Recursion: p_ell = (W^(ell+1))^T/sqrt(m) * D_{ell+1} * p_{ell+1}
            p_next = self._compute_p(ell + 1, alpha_idx)
            sigma_prime_next = self.sigma_derivatives[ell + 1][alpha_idx]
            W_next_T = self.weights[ell].T / jnp.sqrt(self.m) # weights[ell] is W^(ell+1)
            res = W_next_T @ sigma_prime_next @ p_next
        
        self._memo_p[(ell, alpha_idx)] = res
        return res

    def _get_u(self, ell: int, alpha_idx: int) -> jnp.ndarray:
        """Computes u_ell(x_alpha) = D_ell(x_alpha) p_ell(x_alpha)."""
        p_ell = self._compute_p(ell, alpha_idx)
        return self.sigma_derivatives[ell][alpha_idx] @ p_ell

    def kernel2(self, alpha1: int, alpha2: int) -> float:
        """
        Computes the K_2 kernel (NTK) K^(2)(x_alpha1, x_alpha2).
        """
        k2 = 0.0
        # Sum over G^(l) for l=1..H
        for l in range(1, self.H + 1):
            p_l_a1 = self._compute_p(l, alpha1)
            p_l_a2 = self._compute_p(l, alpha2)
            
            # u_l(x) = D_l(x) * p_l(x)
            u_l_a1 = self._get_u(l, alpha1)
            u_l_a2 = self._get_u(l, alpha2)
            
            # G^(l) = <u_l(a1), u_l(a2)> * <x^(l-1)(a1), x^(l-1)(a2)>
            g_l = jnp.dot(u_l_a1, u_l_a2) * jnp.dot(self.feature_maps[l-1][alpha1], self.feature_maps[l-1][alpha2])
            k2 += g_l
            
        # Add G^(H+1) = <x^H(a1), x^H(a2)>
        g_H_plus_1 = jnp.dot(self.feature_maps[self.H][alpha1], self.feature_maps[self.H][alpha2])
        k2 += g_H_plus_1
        
        return k2

    def _compute_x_dot(self, ell: int, alpha_idx: int, beta: int) -> jnp.ndarray:
        """Computes d/dt x^ell(alpha_idx) evaluated along the training path of x_beta."""
        if (ell, alpha_idx, beta) in self._memo_x_dot:
            return self._memo_x_dot[(ell, alpha_idx, beta)]

        if ell == 0:
            # x^0 is input data, its derivative is 0.
            # Assuming x^0 has shape (d,)
            d = self.feature_maps[0][alpha_idx].shape[0]
            return jnp.zeros(d)

        # d/dt x^ell = D^ell * d/dt z^ell
        # d/dt z^ell = 1/sqrt(m) * [ (d/dt W^ell) x^{ell-1} + W^ell (d/dt x^{ell-1}) ]
        
        # Replacement for d/dt W^ell is u_ell(beta) (x^{ell-1}(beta))^T / sqrt(m)
        u_ell_beta = self._get_u(ell, beta)
        x_ell_minus_1_beta = self.feature_maps[ell-1][beta]
        x_ell_minus_1_alpha = self.feature_maps[ell-1][alpha_idx]
        
        # (d/dt W^ell) x^{ell-1}(alpha) part
        term1 = jnp.dot(x_ell_minus_1_beta, x_ell_minus_1_alpha) * u_ell_beta / self.m

        # W^ell (d/dt x^{ell-1}) part
        x_dot_prev = self._compute_x_dot(ell - 1, alpha_idx, beta)
        term2 = self.weights[ell-1] @ x_dot_prev / jnp.sqrt(self.m)
        
        z_dot = term1 + term2
        res = self.sigma_derivatives[ell][alpha_idx] @ z_dot

        self._memo_x_dot[(ell, alpha_idx, beta)] = res
        return res

    def _compute_p_W_replaced(self, ell: int, alpha_idx: int, beta: int, j: int) -> jnp.ndarray:
        """Computes derivative of p_ell w.r.t. W^j replacement."""
        if ell >= j:
            # p_ell does not depend on W^j for ell >= j
            return jnp.zeros_like(self._compute_p(ell, alpha_idx))
        
        if (ell, alpha_idx, beta, j) in self._memo_p_repl_W:
            return self._memo_p_repl_W[(ell, alpha_idx, beta, j)]

        # Recursion for d/dt p_ell
        # d/dt p_ell = (W^{ell+1})^T D^{ell+1} (d/dt p_{ell+1}) + (d/dt W^{ell+1})^T D^{ell+1} p_{ell+1}
        
        # Term from recursive call
        p_dot_next = self._compute_p_W_replaced(ell + 1, alpha_idx, beta, j)
        W_next_T = self.weights[ell].T / jnp.sqrt(self.m)
        sigma_prime_next = self.sigma_derivatives[ell + 1][alpha_idx]
        res = W_next_T @ sigma_prime_next @ p_dot_next

        # Term from d/dt W^{ell+1}
        if ell + 1 == j:
            # replacement for d/dt W^j is u_j(beta) x^{j-1}(beta)^T / m
            u_j_beta = self._get_u(j, beta)
            x_j_minus_1_beta = self.feature_maps[j-1][beta]
            
            p_j_alpha = self._compute_p(j, alpha_idx)
            sigma_prime_j_alpha = self.sigma_derivatives[j][alpha_idx]
            
            # (d/dt W^j)^T D^j p^j
            # u_j(beta) (x^{j-1}(beta)^T D^j p^j) / m
            term = jnp.dot(x_j_minus_1_beta, sigma_prime_j_alpha @ p_j_alpha) * u_j_beta / self.m
            res += term

        self._memo_p_repl_W[(ell, alpha_idx, beta, j)] = res
        return res

    def _compute_p_a_replaced(self, ell: int, alpha_idx: int, beta: int) -> jnp.ndarray:
        """Helper to compute p_ell where 'a' is replaced by x_beta^H."""
        if (ell, alpha_idx, beta) in self._memo_p_repl_a:
            return self._memo_p_repl_a[(ell, alpha_idx, beta)]

        if ell == self.H:
            # Replacement rule: a -> x_beta^H
            res = self.feature_maps[self.H][beta] / jnp.sqrt(self.m)
        else:
            p_next_repl = self._compute_p_a_replaced(ell + 1, alpha_idx, beta)
            sigma_prime_next = self.sigma_derivatives[ell + 1][alpha_idx]
            W_next_T = self.weights[ell].T / jnp.sqrt(self.m)
            res = W_next_T @ sigma_prime_next @ p_next_repl
        
        self._memo_p_repl_a[(ell, alpha_idx, beta)] = res
        return res

    def kernel3(self, alpha1: int, alpha2: int, beta: int) -> float:
        """
        Computes the K_3 kernel K^(3)(x_alpha1, x_alpha2, x_beta).
        This is a more complete implementation.
        """
        k3 = 0
        for l in range(1, self.H + 1):
            u_l_a1 = self._get_u(l, alpha1)
            u_l_a2 = self._get_u(l, alpha2)

            # --- Contribution from d/dt(u) ---
            # 1. Replacement of 'a'
            p_l_a1_repl_a = self._compute_p_a_replaced(l, alpha1, beta)
            p_l_a2_repl_a = self._compute_p_a_replaced(l, alpha2, beta)
            u_l_a1_dot_a = self.sigma_derivatives[l][alpha1] @ p_l_a1_repl_a
            u_l_a2_dot_a = self.sigma_derivatives[l][alpha2] @ p_l_a2_repl_a
            g_l_deriv_u = jnp.dot(u_l_a1_dot_a, u_l_a2) + jnp.dot(u_l_a1, u_l_a2_dot_a)

            # 2. Replacement of W^j for j > l
            for j in range(l + 1, self.H + 1):
                p_l_a1_dot_Wj = self._compute_p_W_replaced(l, alpha1, beta, j)
                p_l_a2_dot_Wj = self._compute_p_W_replaced(l, alpha2, beta, j)
                u_l_a1_dot_Wj = self.sigma_derivatives[l][alpha1] @ p_l_a1_dot_Wj
                u_l_a2_dot_Wj = self.sigma_derivatives[l][alpha2] @ p_l_a2_dot_Wj
                g_l_deriv_u += jnp.dot(u_l_a1_dot_Wj, u_l_a2) + jnp.dot(u_l_a1, u_l_a2_dot_Wj)
            
            x_prod = jnp.dot(self.feature_maps[l-1][alpha1], self.feature_maps[l-1][alpha2])
            k3 += g_l_deriv_u * x_prod

            # --- Contribution from derivative of x^(l-1) ---
            # This is due to W^j for j < l
            x_dot_a1 = self._compute_x_dot(l - 1, alpha1, beta)
            x_dot_a2 = self._compute_x_dot(l - 1, alpha2, beta)
            x_a1 = self.feature_maps[l-1][alpha1]
            x_a2 = self.feature_maps[l-1][alpha2]

            g_l_deriv_x = jnp.dot(x_dot_a1, x_a2) + jnp.dot(x_a1, x_dot_a2)
            k3 += jnp.dot(u_l_a1, u_l_a2) * g_l_deriv_x

        # --- Contribution from derivative of G^(H+1) ---
        x_dot_H_a1 = self._compute_x_dot(self.H, alpha1, beta)
        x_dot_H_a2 = self._compute_x_dot(self.H, alpha2, beta)
        x_H_a1 = self.feature_maps[self.H][alpha1]
        x_H_a2 = self.feature_maps[self.H][alpha2]
        k3 += jnp.dot(x_dot_H_a1, x_H_a2) + jnp.dot(x_H_a1, x_dot_H_a2)
        
        return k3
