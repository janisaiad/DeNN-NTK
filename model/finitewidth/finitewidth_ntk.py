import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Callable, Tuple
import neural_tangents as nt
from functools import partial

class NtkEmpiricalJax:
    """
    Computes the finite width NTK for a fully connected network using manual
    JAX implementation based on the recursive formulas.
    """
    def __init__(self, weights: List[jnp.ndarray],
                 sigma_derivatives: Dict[int, List[jnp.ndarray]],
                 feature_maps: Dict[int, List[jnp.ndarray]]):
        """
        Initializes the NTK computation.

        Args:
            weights: List of weight matrices [W^(1), ..., W^(H), a].
            sigma_derivatives: Dict mapping layer to activation derivatives.
            feature_maps: Dict mapping layer to feature maps.
        """
        self.weights = weights
        self.sigma_derivatives = sigma_derivatives
        self.feature_maps = feature_maps
        self.H = len(weights) - 1
        if not self.weights:
            self.m = 0
        elif self.weights[0].ndim > 1:
            self.m = weights[0].shape[0]
        else:
            self.m = weights[0].shape[0] if weights[0].ndim > 0 else 1
        self.n_inputs = len(feature_maps.get(0, []))

    def _compute_G(self, ell: int, mu: int) -> jnp.ndarray:
        """
        Computes G^(ell)_mu = df(x_mu)/dh^(ell)_mu recursively.
        """
        if ell == self.H:
            return (1/jnp.sqrt(self.m)) * self.sigma_derivatives[self.H][mu] * self.weights[-1]

        G_next = self._compute_G(ell + 1, mu)
        W_next = self.weights[ell].T
        sigma_prime = self.sigma_derivatives[ell][mu]
        
        return (1/jnp.sqrt(self.m)) * sigma_prime * (W_next @ G_next)

    def layer_ntk(self, l: int, alpha: int, beta: int) -> float:
        """
        Computes the NTK for layer l, NTK^(l)(x_alpha, x_beta).
        l is 1-based index, from 1 to H+1.
        """
        if not (1 <= l <= self.H + 1):
            raise ValueError(f"Layer index l must be between 1 and {self.H + 1}")

        if l == self.H + 1:
            x_H_alpha = self.feature_maps[self.H][alpha]
            x_H_beta = self.feature_maps[self.H][beta]
            return (1/self.m) * jnp.dot(x_H_alpha, x_H_beta)

        G_alpha = self._compute_G(l, alpha)
        G_beta = self._compute_G(l, beta)
        
        x_prev_alpha = self.feature_maps[l-1][alpha]
        x_prev_beta = self.feature_maps[l-1][beta]
        
        term1 = jnp.dot(x_prev_alpha, x_prev_beta)
        term2 = jnp.dot(G_alpha, G_beta)
        
        return (1/self.m) * term1 * term2

    def ntk(self, alpha: int, beta: int) -> Tuple[float, Dict[int, float]]:
        """
        Computes the full NTK between x_alpha and x_beta.
        Returns the total NTK and a dictionary with each layer's NTK.
        """
        total_ntk = 0.0
        layer_ntks = {}
        for l in range(1, self.H + 2):
            layer_val = self.layer_ntk(l, alpha, beta)
            total_ntk += layer_val
            layer_ntks[l] = layer_val
        
        return total_ntk, layer_ntks

class NtkEmpiricalNeuralTangent:
    """
    Computes the finite width NTK using the neural-tangents library.
    """
    def __init__(self, weights: List[jnp.ndarray],
                 feature_maps: Dict[int, List[jnp.ndarray]],
                 activation_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Initializes the NTK computation using neural-tangents.

        Args:
            weights: List of weight matrices [W^(1), ..., W^(H), a].
            feature_maps: Dict mapping layer to feature maps (only x^(0) is used).
            activation_fn: The activation function (e.g., jax.nn.relu).
        """
        self.weights = weights
        self.feature_maps = feature_maps
        self.H = len(weights) - 1
        if not self.weights:
            self.m = 0
        elif self.weights[0].ndim > 1:
            self.m = weights[0].shape[0]
        else:
            self.m = weights[0].shape[0] if weights[0].ndim > 0 else 1

        # Define the network application function for neural-tangents
        def apply_fn(params, x):
            hidden = x
            # Hidden layers
            for i in range(self.H):
                W = params[i]
                # scaling from paper
                pre_activations = jnp.dot(W, hidden) / jnp.sqrt(W.shape[1])
                hidden = activation_fn(pre_activations)
            
            # Output layer
            a = params[self.H]
            return jnp.dot(a, hidden) / jnp.sqrt(self.m)

        self.ntk_fn = nt.empirical_ntk_fn(apply_fn)
        
        # For layer-wise NTK, we need to define the network using nt.stax
        layers = []
        for i in range(self.H):
            layers.append(nt.stax.Dense(self.m, W_std=1.0, b_std=0.0))
            layers.append(activation_fn)
        layers.append(nt.stax.Dense(1, W_std=1.0, b_std=0.0)) # Assuming scalar output
        
        _, params_nt = nt.stax.serial(*layers)
        
        # Note: neural-tangents creates its own parameter structure.
        # We can't easily use the passed `weights` for layer-wise decomposition
        # without more complex reconstruction. For now, layer_ntk will be noted as
        # hard to implement with current API.
        
    def ntk(self, alpha: int, beta: int) -> float:
        """
        Computes the full NTK between x_alpha and x_beta.
        """
        x_alpha = self.feature_maps[0][alpha]
        x_beta = self.feature_maps[0][beta]

        # neural-tangents expects batches, so we add a batch dimension.
        x_alpha_batch = jnp.expand_dims(x_alpha, 0)
        x_beta_batch = jnp.expand_dims(x_beta, 0)

        kernel_matrix = self.ntk_fn(x_alpha_batch, x_beta_batch, self.weights)
        return kernel_matrix[0, 0]

    def layer_ntk(self, l: int, alpha: int, beta: int) -> float:
        """
        Computes the NTK for layer l.
        NOTE: This is non-trivial to implement with an arbitrary set of weights
        in neural-tangents without constructing a nt.stax model and matching parameters.
        This is a placeholder for a potential future implementation.
        """
        raise NotImplementedError("Layer-wise NTK with neural-tangents requires a more detailed setup.")