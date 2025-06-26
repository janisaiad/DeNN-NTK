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
    """we plot histogram of eigenvalues distribution for each k and mean eigenvalues vs order"""
    n_eigenvalues = eigenvalues.shape[1]  # number of eigenvalues per experiment
    
    # we compute mean eigenvalues across experiments (excluding the last one)
    mean_eigenvalues = np.mean(eigenvalues[:, :-1], axis=0)  # we exclude last eigenvalue
    eigenvalue_orders = list(range(1, len(mean_eigenvalues) + 1))  # we start from 1
    
    # we create figure with histograms and mean plot
    plt.figure(figsize=(15, 2*n_eigenvalues + 5))
    
    # we plot histograms for each eigenvalue
    for k in range(n_eigenvalues):
        plt.subplot(n_eigenvalues + 1, 2, 2*k + 1)
        plt.hist(eigenvalues[:, k], bins=50, density=True)
        plt.title(f'Eigenvalue {k+1} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
        
        # we add mean line
        mean_val = np.mean(eigenvalues[:, k])
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
    
    # we plot mean eigenvalues vs order (excluding last eigenvalue)
    plt.subplot(n_eigenvalues + 1, 2, 2*n_eigenvalues + 1)
    plt.plot(eigenvalue_orders, mean_eigenvalues, 'o-', linewidth=2, markersize=8, color='blue')
    plt.title(f'Mean Eigenvalues vs Order (Excluding Last)\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvalue Order')
    plt.ylabel('Mean Eigenvalue')
    plt.grid(True)
    plt.yscale('log')  # we use log scale for better visualization
    
    # we add eigenvalue gap visualization
    plt.subplot(n_eigenvalues + 1, 2, 2*n_eigenvalues + 2)
    if len(mean_eigenvalues) > 1:
        eigenvalue_gaps = np.diff(mean_eigenvalues)  # we compute gaps between consecutive eigenvalues
        gap_orders = list(range(1, len(eigenvalue_gaps) + 1))
        plt.plot(gap_orders, eigenvalue_gaps, 'o-', linewidth=2, markersize=8, color='orange')
        plt.title(f'Eigenvalue Gaps vs Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
        plt.xlabel('Gap Order (between k and k+1)')
        plt.ylabel('Eigenvalue Gap')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VALUES, f'eigenvalue_dist_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.show()
    
    # we return the computed means for storage
    return {
        'mean_eigenvalues': mean_eigenvalues.tolist(),  # we convert to list for JSON serialization
        'eigenvalue_orders': eigenvalue_orders,
        'eigenvalue_gaps': np.diff(mean_eigenvalues).tolist() if len(mean_eigenvalues) > 1 else [],
        'config': {'N': N, 'D_IN': D_IN, 'M': M, 'L': L}
    }

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
        'eigenvector_tests': [],
        'last_eigenvector_analysis': {}  # we add analysis for last eigenvector
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
        
        # we analyze last eigenvector specifically
        if k == n_vectors - 1:
            last_eigenvector = vectors_k[-1]  # we take last experiment's last eigenvector
            last_eig_mean = float(np.mean(last_eigenvector))
            last_eig_std = float(np.std(last_eigenvector))
            
            test_results['last_eigenvector_analysis'] = {
                'mean': last_eig_mean,
                'std': last_eig_std,
                'is_constant_threshold': last_eig_std < 1e-10  # we check if nearly constant
            }
            
            print(f"Config N{N}_D{D_IN}_M{M}_L{L} - Last eigenvector mean: {last_eig_mean:.6f}, std: {last_eig_std:.6f}")
    
    # we save test results as JSON
    test_filename = f'test_results_N{N}_D{D_IN}_M{M}_L{L}.json'
    with open(os.path.join(PATH_TO_TESTS, test_filename), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # we create plots with eigenvector order on x-axis
    plt.figure(figsize=(15, 12))
    
    # we extract data for plotting
    eigenvector_indices = [t['index'] + 1 for t in test_results['eigenvector_tests']]  # we start from 1
    max_discrepancies = [t['max_discrepancy'] for t in test_results['eigenvector_tests']]
    ks_stats = [t['ks_stat'] for t in test_results['eigenvector_tests']]
    ks_pvalues = [t['ks_pvalue'] for t in test_results['eigenvector_tests']]
    
    # we plot max discrepancy vs eigenvector order
    plt.subplot(2, 2, 1)
    plt.plot(eigenvector_indices, max_discrepancies, 'o-', linewidth=2, markersize=6)
    plt.title(f'Maximum Discrepancy vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('Max Discrepancy')
    plt.grid(True)
    
    # we plot KS statistic vs eigenvector order
    plt.subplot(2, 2, 2)
    plt.plot(eigenvector_indices, ks_stats, 'o-', color='orange', linewidth=2, markersize=6)
    plt.title(f'KS Statistics vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('KS Statistic')
    plt.grid(True)
    
    # we plot KS p-values vs eigenvector order
    plt.subplot(2, 2, 3)
    plt.plot(eigenvector_indices, ks_pvalues, 'o-', color='green', linewidth=2, markersize=6)
    plt.title(f'KS p-values vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('KS p-value')
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')  # we add significance threshold
    plt.legend()
    plt.grid(True)
    
    # we plot last eigenvector analysis
    plt.subplot(2, 2, 4)
    last_eig_info = test_results['last_eigenvector_analysis']
    bars = plt.bar(['Mean', 'Std'], [last_eig_info['mean'], last_eig_info['std']], 
                   color=['red', 'purple'], alpha=0.7)
    plt.title(f'Last Eigenvector Statistics\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.ylabel('Value')
    plt.grid(True, axis='y')
    
    # we add value labels on bars
    for bar, value in zip(bars, [last_eig_info['mean'], last_eig_info['std']]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'eigenvector_analysis_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.show()
    
    return {
        'avg_max_discrepancy': np.mean([t['max_discrepancy'] for t in test_results['eigenvector_tests']]),
        'avg_ks_stat': np.mean([t['ks_stat'] for t in test_results['eigenvector_tests']]),
        'avg_ks_pvalue': np.mean([t['ks_pvalue'] for t in test_results['eigenvector_tests']]),
        'last_eig_mean': test_results['last_eigenvector_analysis']['mean'],
        'last_eig_std': test_results['last_eigenvector_analysis']['std'],
        'last_eig_is_constant': test_results['last_eigenvector_analysis']['is_constant_threshold'],
        'N': N,
        'D_IN': D_IN,
        'M': M,
        'L': L
    }

# %%
def create_improved_eigenvector_plots(all_results):
    """we create improved plots showing trends across configurations"""
    if not all_results:
        return
        
    n_configs = len(all_results)
    
    # we create plots showing variation across all configurations
    plt.figure(figsize=(20, 15))
    
    # we plot max discrepancy trends
    plt.subplot(3, 2, 1)
    max_disc_values = [r['avg_max_discrepancy'] for r in all_results]
    plt.plot(range(n_configs), max_disc_values, 'o-', label='Average Max Discrepancy')
    plt.title('Maximum Discrepancy Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Max Discrepancy')
    plt.grid(True)
    plt.legend()
    
    # we plot KS statistics trends  
    plt.subplot(3, 2, 2)
    ks_stat_values = [r['avg_ks_stat'] for r in all_results]
    plt.plot(range(n_configs), ks_stat_values, 'o-', label='Average KS Statistic', color='orange')
    plt.title('KS Statistics Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('KS Statistic')
    plt.grid(True)
    plt.legend()
    
    # we plot KS p-values trends
    plt.subplot(3, 2, 3)
    ks_pval_values = [r['avg_ks_pvalue'] for r in all_results]
    plt.plot(range(n_configs), ks_pval_values, 'o-', label='Average KS p-value', color='green')
    plt.title('KS p-values Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('KS p-value')
    plt.grid(True)
    plt.legend()
    
    # we plot last eigenvector mean
    plt.subplot(3, 2, 4)
    last_eig_means = [r['last_eig_mean'] for r in all_results]
    plt.plot(range(n_configs), last_eig_means, 'o-', label='Last Eigenvector Mean', color='red')
    plt.title('Last Eigenvector Mean Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Mean Value')
    plt.grid(True)
    plt.legend()
    
    # we plot last eigenvector std
    plt.subplot(3, 2, 5)
    last_eig_stds = [r['last_eig_std'] for r in all_results]
    plt.plot(range(n_configs), last_eig_stds, 'o-', label='Last Eigenvector Std', color='purple')
    plt.title('Last Eigenvector Standard Deviation Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    plt.legend()
    
    # we plot constant detection
    plt.subplot(3, 2, 6)
    constant_flags = [1 if r['last_eig_is_constant'] else 0 for r in all_results]
    plt.plot(range(n_configs), constant_flags, 'o-', label='Is Constant (1=Yes, 0=No)', color='brown')
    plt.title('Last Eigenvector Constant Detection')
    plt.xlabel('Configuration Index')
    plt.ylabel('Is Constant')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, 'improved_eigenvector_analysis.png'))
    plt.show()

# %%
# we process all files in the data directory
all_results = []
eigenvalue_means_storage = []  # we store eigenvalue means separately
files = [f for f in os.listdir(os.path.join(PATH_TO_DATA, "vectors")) if f.startswith('ntk_eigenvectors_')]

print("Processing all experiment files...")
print("=" * 50)
for file in tqdm(files):
    try:
        # we extract configuration from filename
        N, D_IN, M, L = get_config_from_filename(file)
        
        # we load and analyze data
        eigenvalues_data, eigenvectors_data = load_experiment_data(N, D_IN, M, L)
        
        # we analyze distributions and get eigenvalue means
        eigenvalue_results = plot_eigenvalue_distribution(eigenvalues_data['eigenvalues'], N, D_IN, M, L)
        eigenvector_results = analyze_eigenvector_distribution(eigenvectors_data['eigenvectors'], N, D_IN, M, L)
        
        # we combine results
        combined_results = {**eigenvector_results, **eigenvalue_results}
        all_results.append(combined_results)
        eigenvalue_means_storage.append(eigenvalue_results)
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("=" * 50)
print("LAST EIGENVECTOR ANALYSIS SUMMARY:")
print("=" * 50)
for result in all_results:
    config_str = f"N{result['N']}_D{result['D_IN']}_M{result['M']}_L{result['L']}"
    constant_str = "CONSTANT" if result['last_eig_is_constant'] else "VARIABLE"
    print(f"{config_str}: mean={result['last_eig_mean']:.6f}, std={result['last_eig_std']:.6f} [{constant_str}]")

print("\n" + "=" * 50)
print("EIGENVALUE MEANS SUMMARY:")
print("=" * 50)
for eig_result in eigenvalue_means_storage:
    config = eig_result['config']
    config_str = f"N{config['N']}_D{config['D_IN']}_M{config['M']}_L{config['L']}"
    n_eigenvals = len(eig_result['mean_eigenvalues'])
    print(f"{config_str}: {n_eigenvals} eigenvalues (excluding last)")
    for i, mean_val in enumerate(eig_result['mean_eigenvalues'][:5]):  # we show first 5
        print(f"  Eigenvalue {i+1}: {mean_val:.6f}")
    if n_eigenvals > 5:
        print(f"  ... and {n_eigenvals-5} more")

# we create improved plots
create_improved_eigenvector_plots(all_results)

# we save results to CSV using native Python
if all_results:
    fieldnames = all_results[0].keys()
    with open(os.path.join(PATH_TO_STATS, 'all_results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

# we save eigenvalue means separately
with open(os.path.join(PATH_TO_STATS, 'eigenvalue_means.json'), 'w') as f:
    json.dump(eigenvalue_means_storage, f, indent=2)

# we save results to JSON
results_json = {
    'metadata': {
        'total_experiments': len(all_results),
        'parameters_tested': ['L', 'N', 'M', 'D_IN'],
        'last_eigenvector_summary': {
            'constant_count': sum(1 for r in all_results if r['last_eig_is_constant']),
            'variable_count': sum(1 for r in all_results if not r['last_eig_is_constant']),
            'mean_of_means': np.mean([r['last_eig_mean'] for r in all_results]),
            'mean_of_stds': np.mean([r['last_eig_std'] for r in all_results])
        },
        'eigenvalue_analysis': {
            'total_configs_analyzed': len(eigenvalue_means_storage),
            'average_num_eigenvalues': np.mean([len(e['mean_eigenvalues']) for e in eigenvalue_means_storage])
        }
    },
    'results': all_results,
    'eigenvalue_means': eigenvalue_means_storage
}
with open(os.path.join(PATH_TO_STATS, 'all_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# we plot trends with respect to each parameter using native Python grouping
params = ['L', 'N', 'M', 'D_IN']
metrics = ['avg_max_discrepancy', 'avg_ks_stat', 'avg_ks_pvalue', 'last_eig_mean', 'last_eig_std']  # we add new metrics

for param in params:
    plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        
        # we group by parameter manually and compute mean
        grouped_data = defaultdict(list)
        for result in all_results:
            grouped_data[result[param]].append(result[metric])
        
        # we compute means for each parameter value
        param_values = sorted(grouped_data.keys())
        mean_values = [np.mean(grouped_data[p]) for p in param_values]
        std_values = [np.std(grouped_data[p]) for p in param_values]  # we add error bars
        
        plt.errorbar(param_values, mean_values, yerr=std_values, fmt='o-', capsize=5)
        
        plt.title(f'{metric} vs {param}')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'trends_wrt_{param}.png'))
    plt.show()

print("\nAnalysis complete. Results saved to:")
print(f"- Statistics (CSV & JSON): {PATH_TO_STATS}")
print(f"- Vector plots: {PATH_TO_PLOTS_VECTORS}")
print(f"- Value plots: {PATH_TO_PLOTS_VALUES}")
print(f"- Test results: {PATH_TO_TESTS}")
print(f"- Last eigenvector summary: See JSON metadata section")
