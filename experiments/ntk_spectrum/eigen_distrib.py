# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis of NTK Eigenvalue and Eigenvector Distributions
#
# We analyze the distributions of eigenvalues and eigenvectors of the NTK matrix.

# %%
import os
import numpy as np
import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv  # we use csv instead of pandas
from collections import defaultdict  # we use defaultdict for grouping
import json  # we add json for storing test results

# %%
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")
PATH_TO_STATS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "stats")
PATH_TO_PLOTS_VECTORS = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "vectors")
PATH_TO_PLOTS_VALUES = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "values")
PATH_TO_TESTS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "tests")  # we add path for test results

# we create necessary directories
for path in [PATH_TO_STATS, PATH_TO_PLOTS_VECTORS, PATH_TO_PLOTS_VALUES, PATH_TO_TESTS]:
    os.makedirs(path, exist_ok=True)

# %%
def get_config_from_filename(filename):
    """we extract configuration parameters from filename"""
    parts = filename.replace(".npy", "").split("_")
    N = int(parts[-4][1:])
    D = int(parts[-3][1:])
    M = int(parts[-2][1:])
    L = int(parts[-1][1:])
    return N, D, M, L

def load_experiment_data(N, D_IN, M, L):
    """we load eigenvalues and eigenvectors data for a specific configuration"""
    filename_eigenvalues = f"values/ntk_eigenvalues_N{N}_D{D_IN}_M{M}_L{L}.npy"
    filename_eigenvectors = f"vectors/ntk_eigenvectors_N{N}_D{D_IN}_M{M}_L{L}.npy"
    
    eigenvalues_data = np.load(os.path.join(PATH_TO_DATA, filename_eigenvalues), allow_pickle=True).item()
    eigenvectors_data = np.load(os.path.join(PATH_TO_DATA, filename_eigenvectors), allow_pickle=True).item()
    
    return eigenvalues_data, eigenvectors_data

# %%
def plot_eigenvalue_distribution(eigenvalues, N, D_IN, M, L):
    """we plot histogram of eigenvalues distribution for each k"""
    n_eigenvalues = eigenvalues.shape[1]  # number of eigenvalues per experiment
    
    plt.figure(figsize=(10, 2*n_eigenvalues))
    for k in range(n_eigenvalues):
        plt.subplot(n_eigenvalues, 1, k+1)
        plt.hist(eigenvalues[:, k], bins=50, density=True)
        plt.title(f'Eigenvalue {k+1} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VALUES, f'eigenvalue_dist_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()

# %%
def compute_max_discrepancy(vectors):
    """we compute maximum discrepancy as uniformity measure on sphere"""
    n_vectors = vectors.shape[0]
    max_discrepancy = 0
    
    for i in range(n_vectors):
        for j in range(i+1, n_vectors):
            dist = jnp.abs(jnp.dot(vectors[i], vectors[j]))
            max_discrepancy = max(max_discrepancy, dist)
            
    return float(max_discrepancy)  # we convert to float for JSON serialization

# %%
def test_uniformity_ks(vectors, n_projections=1000):
    """we perform Kolmogorov-Smirnov test for uniformity"""
    random_directions = np.random.randn(n_projections, vectors.shape[1])
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
    
    projections = np.dot(vectors, random_directions.T)
    
    ks_stats = []
    p_values = []
    for proj in projections.T:
        ks_stat, p_value = stats.kstest(proj, 'uniform', args=(-1, 2))
        ks_stats.append(float(ks_stat))  # we convert to float for JSON serialization
        p_values.append(float(p_value))  # we convert to float for JSON serialization
        
    return np.mean(ks_stats), np.mean(p_values)

# %%
def analyze_eigenvector_distribution(eigenvectors, N, D_IN, M, L):
    """we analyze eigenvector distribution for each eigenspace"""
    n_vectors = eigenvectors.shape[1]
    
    test_results = {
        'config': {
            'N': N,
            'D_IN': D_IN,
            'M': M,
            'L': L
        },
        'eigenvector_tests': []
    }
    
    for k in tqdm(range(n_vectors)):
        vectors_k = eigenvectors[:, :, k]
        
        max_disc = compute_max_discrepancy(vectors_k)
        ks_stat, p_value = test_uniformity_ks(vectors_k)
        
        test_results['eigenvector_tests'].append({
            'index': k,
            'max_discrepancy': max_disc,
            'ks_stat': ks_stat,
            'ks_pvalue': p_value
        })
    
    # we save test results as JSON
    test_filename = f'test_results_N{N}_D{D_IN}_M{M}_L{L}.json'
    with open(os.path.join(PATH_TO_TESTS, test_filename), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # we create vertical visualization for each eigenvector
    plt.figure(figsize=(15, 5*n_vectors))
    
    for k in range(n_vectors):
        plt.subplot(n_vectors, 3, 3*k + 1)
        plt.plot([test_results['eigenvector_tests'][k]['max_discrepancy']], 'o')
        plt.title(f'Maximum Discrepancy (Eigenvector {k+1})')
        plt.grid(True)
        
        plt.subplot(n_vectors, 3, 3*k + 2)
        plt.plot([test_results['eigenvector_tests'][k]['ks_stat']], 'o')
        plt.title(f'KS Statistics (Eigenvector {k+1})')
        plt.grid(True)
        
        plt.subplot(n_vectors, 3, 3*k + 3)
        plt.plot([test_results['eigenvector_tests'][k]['ks_pvalue']], 'o')
        plt.title(f'KS p-values (Eigenvector {k+1})')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'eigenvector_analysis_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()
    
    return {
        'avg_max_discrepancy': np.mean([t['max_discrepancy'] for t in test_results['eigenvector_tests']]),
        'avg_ks_stat': np.mean([t['ks_stat'] for t in test_results['eigenvector_tests']]),
        'avg_ks_pvalue': np.mean([t['ks_pvalue'] for t in test_results['eigenvector_tests']]),
        'N': N,
        'D_IN': D_IN,
        'M': M,
        'L': L
    }

# %%
# we process all files in the data directory
all_results = []
files = [f for f in os.listdir(os.path.join(PATH_TO_DATA, "vectors")) if f.startswith('ntk_eigenvectors_')]

print("Processing all experiment files...")
for file in tqdm(files):
    try:
        # we extract configuration from filename
        N, D_IN, M, L = get_config_from_filename(file)
        
        # we load and analyze data
        eigenvalues_data, eigenvectors_data = load_experiment_data(N, D_IN, M, L)
        
        # we analyze distributions
        plot_eigenvalue_distribution(eigenvalues_data['eigenvalues'], N, D_IN, M, L)
        results = analyze_eigenvector_distribution(eigenvectors_data['eigenvectors'], N, D_IN, M, L)
        all_results.append(results)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# we save results to CSV using native Python
if all_results:
    fieldnames = all_results[0].keys()
    with open(os.path.join(PATH_TO_STATS, 'all_results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

# we save results to JSON
results_json = {
    'metadata': {
        'total_experiments': len(all_results),
        'parameters_tested': ['L', 'N', 'M', 'D_IN']
    },
    'results': all_results
}
with open(os.path.join(PATH_TO_STATS, 'all_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# we plot trends with respect to each parameter using native Python grouping
params = ['L', 'N', 'M', 'D_IN']
metrics = ['avg_max_discrepancy', 'avg_ks_stat', 'avg_ks_pvalue']

for param in params:
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        
        # we group by parameter manually and compute mean
        grouped_data = defaultdict(list)
        for result in all_results:
            grouped_data[result[param]].append(result[metric])
        
        # we compute means for each parameter value
        param_values = sorted(grouped_data.keys())
        mean_values = [np.mean(grouped_data[p]) for p in param_values]
        
        plt.plot(param_values, mean_values, 'o-')
        
        plt.title(f'{metric} vs {param}')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'trends_wrt_{param}.png'))
    plt.close()

print("\nAnalysis complete. Results saved to:")
print(f"- Statistics (CSV & JSON): {PATH_TO_STATS}")
print(f"- Vector plots: {PATH_TO_PLOTS_VECTORS}")
print(f"- Value plots: {PATH_TO_PLOTS_VALUES}")
print(f"- Test results: {PATH_TO_TESTS}")
