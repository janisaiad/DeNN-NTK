import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
import os
import neural_tangents as nt
from neural_tangents import stax

N = 100  # Number of data points
D_IN = 10  # Input dimension
M = 100  # Width of the network
L_VALUES = jnp.linspace(10, 200, 20).astype(int)
RANDOM_SEED = 42
N_experiments = 10  # For eigenvalue mean computation

def generate_data(key, n_samples, n_features):
    """Generates random data and normalizes it."""
    data = random.normal(key, (n_samples, n_features))
    norm = jnp.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

def main():
    """Main function to run the experiment."""
    key = random.PRNGKey(RANDOM_SEED)
    key, data_key = random.split(key)
    
    data = generate_data(data_key, N, D_IN)
    eigenvalues_per_L = []
    
    print("Starting NTK eigenvalue analysis using neural-tangents...")
    for L in L_VALUES:
        print(f"Computing for L = {L} layers...")
        key, model_key = random.split(key)
        
        layers = []
        layers.append(stax.Dense(M, W_std=1.0, b_std=0.0, parameterization='ntk'))
        layers.append(stax.Erf())
        
        for _ in range(L - 1):
            layers.append(stax.Dense(M, W_std=1.0, b_std=0.0, parameterization='ntk'))
            layers.append(stax.Erf())
            
        layers.append(stax.Dense(1, W_std=1.0, b_std=0.0, parameterization='ntk'))
        
        init_fn, apply_fn, _ = stax.serial(*layers)
        params = init_fn(model_key, data.shape)[1]

        ntk_fn = nt.empirical_ntk_fn(apply_fn)
        k2_matrix = ntk_fn(data, None, params)
        
        eigenvalues = jnp.linalg.eigvalsh(k2_matrix)
        eigenvalues_per_L.append(eigenvalues)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot eigenvalue histograms
    n_L = len(L_VALUES)
    fig, axes = plt.subplots(1, n_L, figsize=(5 * n_L, 4), sharey=True)
    fig.suptitle(f'Histogram of Empirical NTK Eigenvalues (width M={M})')
    for i, L in enumerate(L_VALUES):
        ax = axes[i]
        ax.hist(eigenvalues_per_L[i], bins='auto', density=True)
        ax.set_title(f"Depth L = {L}")
        ax.set_xlabel("Eigenvalue")
        ax.set_yscale('log')
    axes[0].set_ylabel("Density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/ntk_empirical_eigenvalue_histograms.png")
    print("Saved eigenvalue histograms to plots/ntk_empirical_eigenvalue_histograms.png")
    
    # Plot k-th eigenvalue vs L
    eigenvalues_per_L = np.array(eigenvalues_per_L)
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top 5 NTK Eigenvalues vs Network Depth (width M={M})")
    for k in range(1, min(2*D_IN,N)):
        plt.plot(L_VALUES, eigenvalues_per_L[:, -k], 'o-', label=f'λ_{k}')
    
    plt.plot(L_VALUES, eigenvalues_per_L[:, 0], 'o--', label=f'λ_min')

    plt.xlabel("Number of Layers (L)")
    plt.ylabel("Eigenvalue (log scale)")
    plt.yscale('log')
    plt.xticks(L_VALUES)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plots/kth_eigenvalue_vs_L_empirical.png")
    print("Saved k-th eigenvalue plot to plots/kth_eigenvalue_vs_L_empirical.png")


if __name__ == '__main__':
    main() 