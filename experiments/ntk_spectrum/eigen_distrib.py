import os
import numpy as np
import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm  # we use tqdm for progress bars

import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")

def load_experiment_data(N, D_IN, M, L):
    """we load eigenvalues and eigenvectors data for a specific configuration"""
    filename_eigenvalues = f"ntk_eigenvalues_N{N}_D{D_IN}_M{M}_L{L}.npy"
    filename_eigenvectors = f"ntk_eigenvectors_N{N}_D{D_IN}_M{M}_L{L}.npy"
    
    eigenvalues_data = np.load(os.path.join(PATH_TO_DATA, filename_eigenvalues), allow_pickle=True).item()
    eigenvectors_data = np.load(os.path.join(PATH_TO_DATA, filename_eigenvectors), allow_pickle=True).item()
    
    return eigenvalues_data, eigenvectors_data

def plot_eigenvalue_distribution(eigenvalues, N, D_IN, M, L):
    """we plot histogram of eigenvalues distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues.flatten(), bins=50, density=True)  # we flatten all experiments
    plt.title(f'Eigenvalue Distribution (N={N}, D={D_IN}, M={M}, L={L})')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(os.path.join(PATH_TO_DATA, f'eigenvalue_dist_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()

def compute_max_discrepancy(vectors):
    """we compute maximum discrepancy as uniformity measure on sphere"""
    n_vectors = vectors.shape[0]
    max_discrepancy = 0
    
    # we compute pairwise distances
    for i in range(n_vectors):
        for j in range(i+1, n_vectors):
            dist = jnp.abs(jnp.dot(vectors[i], vectors[j]))
            max_discrepancy = max(max_discrepancy, dist)
            
    return max_discrepancy

def test_uniformity_ks(vectors, n_projections=1000):
    """we perform Kolmogorov-Smirnov test for uniformity"""
    # we generate random directions for projection
    random_directions = np.random.randn(n_projections, vectors.shape[1])
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
    
    # we project vectors onto random directions
    projections = np.dot(vectors, random_directions.T)
    
    # we test against uniform distribution on [-1,1]
    ks_stats = []
    p_values = []
    for proj in projections.T:
        ks_stat, p_value = stats.kstest(proj, 'uniform', args=(-1, 2))
        ks_stats.append(ks_stat)
        p_values.append(p_value)
        
    return np.mean(ks_stats), np.mean(p_values)

def analyze_eigenvector_distribution(eigenvectors, N, D_IN, M, L):
    """we analyze eigenvector distribution for each eigenspace"""
    n_experiments = eigenvectors.shape[0]
    n_vectors = eigenvectors.shape[1]
    
    results = {
        'max_discrepancy': [],
        'ks_stats': [],
        'ks_pvalues': []
    }
    
    for k in tqdm(range(n_vectors)):  # we analyze each eigenvector separately
        vectors_k = eigenvectors[:, :, k]  # shape (n_experiments, N)
        
        # we compute uniformity measures
        max_disc = compute_max_discrepancy(vectors_k)
        ks_stat, p_value = test_uniformity_ks(vectors_k)
        
        results['max_discrepancy'].append(max_disc)
        results['ks_stats'].append(ks_stat)
        results['ks_pvalues'].append(p_value)
    
    # we plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(results['max_discrepancy'])
    plt.title('Maximum Discrepancy')
    plt.xlabel('Eigenvector Index')
    
    plt.subplot(132)
    plt.plot(results['ks_stats'])
    plt.title('KS Statistics')
    plt.xlabel('Eigenvector Index')
    
    plt.subplot(133)
    plt.plot(results['ks_pvalues'])
    plt.title('KS p-values')
    plt.xlabel('Eigenvector Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_DATA, f'eigenvector_analysis_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()
    
    return results

# we analyze first configuration from eigenvectors.py
N, D_IN, M, L = 8, 20, 10, 2  # we use first configuration values

eigenvalues_data, eigenvectors_data = load_experiment_data(N, D_IN, M, L)

# we plot eigenvalue distribution
plot_eigenvalue_distribution(eigenvalues_data['eigenvalues'], N, D_IN, M, L)

# we analyze eigenvector distribution
results = analyze_eigenvector_distribution(eigenvectors_data['eigenvectors'], N, D_IN, M, L)

# we print summary statistics
print("\nSummary Statistics:")
print(f"Average Max Discrepancy: {np.mean(results['max_discrepancy']):.4f}")
print(f"Average KS Statistic: {np.mean(results['ks_stats']):.4f}")
print(f"Average KS p-value: {np.mean(results['ks_pvalues']):.4f}")
