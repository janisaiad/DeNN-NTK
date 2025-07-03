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

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv  # we use csv instead of pandas
from collections import defaultdict  # we use defaultdict for grouping
import json  # we add json for storing test results
from scipy.optimize import least_squares  # we add for brody fit
from scipy.special import gamma as gamma_func  # we add for brody distribution

# %%
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")
PATH_TO_STATS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "stats")
PATH_TO_PLOTS_VECTORS = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "vectors")
PATH_TO_PLOTS_VALUES = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "values")
PATH_TO_PLOTS_SPACING = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "spacing")
PATH_TO_TESTS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "tests")  # we add path for test results

# we create necessary directories
for path in [PATH_TO_STATS, PATH_TO_PLOTS_VECTORS, PATH_TO_PLOTS_VALUES, PATH_TO_TESTS, PATH_TO_PLOTS_SPACING]:
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
    
    # we plot histograms for each eigenvalue with progress tracking
    for k in tqdm(range(n_eigenvalues), desc=f"Plotting eigenvalue distributions for N{N}_D{D_IN}_M{M}_L{L}", leave=False):
        plt.subplot(n_eigenvalues + 1, 2, 2*k + 1)
        plt.hist(eigenvalues[:, k], bins='auto', density=True)
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
    plt.close()
    
    # we return the computed means for storage
    return {
        'mean_eigenvalues': mean_eigenvalues.tolist(),  # we convert to list for JSON serialization
        'eigenvalue_orders': eigenvalue_orders,
        'eigenvalue_gaps': np.diff(mean_eigenvalues).tolist() if len(mean_eigenvalues) > 1 else [],
        'config': {'N': N, 'D_IN': D_IN, 'M': M, 'L': L}
    }

# %%
def analyze_eigenvalue_spacing(eigenvalues, N, D_IN, M, L, output_dir):
    """
    i analyze the distribution of the ratio of consecutive eigenvalue spacings
    and compare it to theoretical distributions from random matrix theory (rmt).
    i also fit the brody distribution to the unfolded spacings.
    """
    all_spacing_ratios = []
    all_unfolded_spacings = []
    # i iterate over each experiment's eigenvalues
    for i in range(eigenvalues.shape[0]):
        # i get eigenvalues, sort them, and filter out near-zero values
        eigs = np.sort(eigenvalues[i, :])
        eigs = eigs[eigs > 1e-12]  # we filter tiny eigenvalues to handle degeneracies

        # we need at least 3 eigenvalues for one ratio, and one more for the bulk
        if len(eigs) < 4:
            continue
        
        # i take the bulk (excluding the largest)
        eigs_bulk = eigs[:-1]
        
        # i compute spacings between adjacent eigenvalues
        spacings = np.diff(eigs_bulk)
        
        # i filter out zero or negative spacings to avoid division errors
        spacings = spacings[spacings > 1e-12]
        
        if len(spacings) < 2:
            continue
            
        # i compute the ratio of consecutive spacings
        ratios = spacings[1:] / spacings[:-1]
        all_spacing_ratios.extend(ratios)

        # we perform unfolding by dividing by the mean spacing
        mean_spacing = np.mean(spacings)
        if mean_spacing > 1e-9:
            unfolded_spacings = spacings / mean_spacing
            all_unfolded_spacings.extend(unfolded_spacings)

    if not all_spacing_ratios:
        print(f"Warning: No spacing ratios computed for config N{N}_D{D_IN}_M{M}_L{L}.")
        return

    all_spacing_ratios = np.array(all_spacing_ratios)
    
    # i compute the mean of r and the mean of min(r, 1/r) for statistical comparison
    mean_r = np.mean(all_spacing_ratios)
    mean_min_r = np.mean(np.minimum(all_spacing_ratios, 1/all_spacing_ratios))

    # i create the plot for spacing ratios
    plt.figure(figsize=(12, 8))
    
    # i plot the histogram of the empirical distribution, capped for visibility
    plt.hist(all_spacing_ratios, bins=100, density=True, label='Empirical Ratios', range=(0, 10), color='skyblue')
    
    # i define theoretical distributions for the ratio r
    r = np.linspace(0, 10, 400)
    # for poissonian spectra (uncorrelated eigenvalues)
    p_poisson = 1 / (1 + r)**2
    # for goe (real-symmetric matrices from wigner-dyson)
    p_goe = (27 / 8) * (r + r**2) / (1 + r + r**2)**2.5
    # for gue (complex hermitian matrices)
    p_gue = (81 * np.sqrt(3) / (4 * np.pi)) * (r + r**2)**2 / (1 + r + r**2)**4
    # for gse (quaternion self-dual matrices)
    p_gse = (729 * np.sqrt(3) / (4 * np.pi)) * (r + r**2)**4 / (1 + r + r**2)**7

    plt.plot(r, p_poisson, 'g--', linewidth=2, label=f'Poisson (Uncorrelated)')
    plt.plot(r, p_goe, 'r-', linewidth=2, label=f'GOE (Wigner-Dyson)')
    plt.plot(r, p_gue, 'b-.', linewidth=2, label=f'GUE')
    plt.plot(r, p_gse, 'm:', linewidth=2, label=f'GSE')
    
    plt.title(f'Consecutive Spacing Ratio Distribution\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Ratio of Consecutive Spacings (r)')
    plt.ylabel('Probability Density P(r)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # i add a text box with statistics
    mean_min_r_poisson = np.log(2)  # theoretical value for poisson
    mean_min_r_goe = 0.535  # theoretical value for goe
    mean_min_r_gue = 0.5996 # theoretical value for gue
    mean_min_r_gse = 0.676 # theoretical value for gse
    
    stats_text = (
        f"Empirical <min(r, 1/r)> = {mean_min_r:.4f}\n\n"
        f"Theoretical <min(r, 1/r)>:\n"
        f"  Poisson: {mean_min_r_poisson:.4f}\n"
        f"  GOE: {mean_min_r_goe:.4f}\n"
        f"  GUE: {mean_min_r_gue:.4f}\n"
        f"  GSE: {mean_min_r_gse:.4f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    # i save the plot
    plot_filename = os.path.join(output_dir, f'consecutive_spacing_ratio_dist_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"Consecutive Spacing Ratio analysis for N{N}_D{D_IN}_M{M}_L{L} complete. Mean ratio <min(r, 1/r)> = {mean_min_r:.4f}")

    # we now analyze the unfolded spacings with brody distribution
    if all_unfolded_spacings:
        all_unfolded_spacings = np.array(all_unfolded_spacings)
        
        # we define brody distribution
        def brody_dist(s, q):
            alpha = (gamma_func((q + 2) / (q + 1)))**(q + 1)
            return (q + 1) * alpha * (s**q) * np.exp(-alpha * s**(q + 1))

        # we define residuals for fitting
        def residuals(q_param, s_data, hist_data):
            # we need to compute the pdf on the bin centers
            bin_centers = (s_data[1:] + s_data[:-1]) / 2
            theoretical_pdf = brody_dist(bin_centers, q_param[0])
            return theoretical_pdf - hist_data
            
        # we compute histogram of unfolded spacings
        hist, bin_edges = np.histogram(all_unfolded_spacings, bins=50, range=(0, 4), density=True)
        
        # we fit brody parameter q
        try:
            # we use least squares to find the best q
            result = least_squares(residuals, x0=[0.5], bounds=(0, 2), args=(bin_edges, hist))
            q_fit = result.x[0]
        except Exception as e:
            print(f"Could not fit Brody distribution for N{N}_D{D_IN}_M{M}_L{L}. Error: {e}")
            q_fit = -1 # we indicate fit failed

        # we create the plot for brody fit
        plt.figure(figsize=(12, 8))
        plt.hist(all_unfolded_spacings, bins=50, density=True, label='Empirical Unfolded Spacings', range=(0, 4), color='skyblue')

        s = np.linspace(0, 4, 400)
        p_poisson_s = np.exp(-s) # p(s) for poisson
        p_goe_s = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4) # p(s) for goe (wigner surmise)

        plt.plot(s, p_poisson_s, 'g--', linewidth=2, label='Poisson (q=0)')
        plt.plot(s, p_goe_s, 'r-', linewidth=2, label='GOE (q=1)')
        
        if q_fit != -1:
            p_brody_fit = brody_dist(s, q_fit)
            plt.plot(s, p_brody_fit, 'k--', linewidth=2, label=f'Brody Fit (q={q_fit:.3f})')
        
        plt.title(f'Unfolded Eigenvalue Spacing Distribution\nConfig N{N}_D{D_IN}_M{M}_L{L}')
        plt.xlabel('Unfolded Spacing (s)')
        plt.ylabel('Probability Density P(s)')
        plt.legend()
        plt.grid(True, alpha=0.5)

        brody_plot_filename = os.path.join(output_dir, f'brody_fit_N{N}_D{D_IN}_M{M}_L{L}.png')
        plt.savefig(brody_plot_filename, dpi=120, bbox_inches='tight')
        plt.close()

    return {
        'mean_spacing_ratio': mean_r,
        'mean_min_spacing_ratio': mean_min_r,
        'config': {'N': N, 'D_IN': D_IN, 'M': M, 'L': L}
    }

# %%
def compute_pca_uniformity(vectors):
    """we use PCA/SVD to test uniformity - uniform distribution should have equal singular values"""
    n_vectors, dimension = vectors.shape
    
    # we center the vectors (subtract mean)
    centered_vectors = vectors - np.mean(vectors, axis=0)
    
    # we compute SVD of the centered matrix
    U, singular_values, Vt = np.linalg.svd(centered_vectors, full_matrices=False)
    
    # we normalize singular values by the largest one
    normalized_svd = singular_values / singular_values[0] if singular_values[0] > 0 else singular_values
    
    # we compute cumulative sum of explained variance
    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative_variance = np.concatenate([[0], np.cumsum(explained_variance)])  # we start with 0
    
    # we compute uniformity metrics
    # 1. Entropy of normalized singular values (high = uniform, low = concentrated)
    eps = 1e-10  # we avoid log(0)
    svd_entropy = -np.sum(explained_variance * np.log(explained_variance + eps))
    max_entropy = np.log(len(singular_values))  # we get maximum possible entropy
    normalized_entropy = svd_entropy / max_entropy if max_entropy > 0 else 0
    
    # 2. Effective rank (number of significant singular values)
    effective_rank = np.exp(svd_entropy)
    
    # 3. Participation ratio (inverse of sum of squared normalized singular values)
    participation_ratio = 1.0 / np.sum(explained_variance ** 2)
    
    return {
        'singular_values': singular_values.tolist(),
        'normalized_svd': normalized_svd.tolist(),
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'svd_entropy': float(svd_entropy),
        'normalized_entropy': float(normalized_entropy),
        'effective_rank': float(effective_rank),
        'participation_ratio': float(participation_ratio),
        'uniformity_score': float(normalized_entropy)  # we use normalized entropy as main uniformity measure
    }

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
    
    # we use tqdm with more descriptive description
    for k in tqdm(range(n_vectors), desc=f"Analyzing eigenvectors for N{N}_D{D_IN}_M{M}_L{L}"):
        vectors_k = eigenvectors[:, :, k]
        
        pca_results = compute_pca_uniformity(vectors_k)
        
        test_results['eigenvector_tests'].append({
            'index': k,
            'pca_uniformity': pca_results
        })
        
        # we analyze last eigenvector specifically
        if k == n_vectors - 1:
            # vectors_k has shape (n_experiments, dimension) for the last eigenvector
            # we compute the barycenter (mean vector) across all experiments
            last_eigenvector_barycenter = np.mean(vectors_k, axis=0)  # we compute barycenter across experiments
            last_eig_mean_components = float(np.mean(last_eigenvector_barycenter))  # we get mean of components
            last_eig_std_components = float(np.std(last_eigenvector_barycenter))  # we get std of components
            
            # we also compute variance across experiments (how similar are the last eigenvectors)
            last_eig_experiment_variance = float(np.mean(np.var(vectors_k, axis=1)))  # we get variance across experiments
            
            test_results['last_eigenvector_analysis'] = {
                'barycenter_mean_component': last_eig_mean_components,
                'barycenter_std_component': last_eig_std_components,
                'experiment_variance': last_eig_experiment_variance,
                'barycenter_vector': last_eigenvector_barycenter.tolist(),  # we store the actual barycenter
                'is_constant_threshold': last_eig_std_components < 1e-10  # we check if barycenter is nearly constant
            }
            
            print(f"Config N{N}_D{D_IN}_M{M}_L{L} - Last eigenvector barycenter mean: {last_eig_mean_components:.6f}, std: {last_eig_std_components:.6f}, exp_var: {last_eig_experiment_variance:.6f}")
            
            # we print the complete barycenter vector explicitly
            print(f"LAST EIGENVECTOR BARYCENTER for N{N}_D{D_IN}_M{M}_L{L}:")
            print(f"Vector dimension: {len(last_eigenvector_barycenter)}")
            print("Complete vector components:")
            for i, component in enumerate(last_eigenvector_barycenter):
                print(f"  [{i:3d}]: {component:+.8f}")
            print(f"Vector L2 norm: {np.linalg.norm(last_eigenvector_barycenter):.8f}")
            print("-" * 60)
    
    # we save test results as JSON
    test_filename = f'test_results_N{N}_D{D_IN}_M{M}_L{L}.json'
    with open(os.path.join(PATH_TO_TESTS, test_filename), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Creating eigenvector plots for N{N}_D{D_IN}_M{M}_L{L}...")
    # we create plots with eigenvector order on x-axis
    plt.figure(figsize=(15, 12))
    
    # we extract data for plotting
    eigenvector_indices = [t['index'] + 1 for t in test_results['eigenvector_tests']]  # we start from 1
    uniformity_scores = [t['pca_uniformity']['uniformity_score'] for t in test_results['eigenvector_tests']]
    effective_ranks = [t['pca_uniformity']['effective_rank'] for t in test_results['eigenvector_tests']]
    participation_ratios = [t['pca_uniformity']['participation_ratio'] for t in test_results['eigenvector_tests']]
    
    # we plot PCA uniformity score vs eigenvector order
    plt.subplot(2, 2, 1)
    plt.plot(eigenvector_indices, uniformity_scores, 'o-', linewidth=2, markersize=6)
    plt.title(f'PCA Uniformity Score vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('Uniformity Score (0=clustered, 1=uniform)')
    plt.grid(True)
    
    # we plot effective rank vs eigenvector order
    plt.subplot(2, 2, 2)
    plt.plot(eigenvector_indices, effective_ranks, 'o-', color='orange', linewidth=2, markersize=6)
    plt.title(f'Effective Rank vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('Effective Rank')
    plt.grid(True)
    
    # we plot participation ratio vs eigenvector order
    plt.subplot(2, 2, 3)
    plt.plot(eigenvector_indices, participation_ratios, 'o-', color='green', linewidth=2, markersize=6)
    plt.title(f'Participation Ratio vs Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvector Order')
    plt.ylabel('Participation Ratio')
    plt.grid(True)
    
    # we plot last eigenvector analysis
    plt.subplot(2, 2, 4)
    last_eig_info = test_results['last_eigenvector_analysis']
    bars = plt.bar(['Barycenter\nMean', 'Barycenter\nStd', 'Experiment\nVariance'], 
                   [last_eig_info['barycenter_mean_component'], 
                    last_eig_info['barycenter_std_component'],
                    last_eig_info['experiment_variance']], 
                   color=['red', 'purple', 'orange'], alpha=0.7)
    plt.title(f'Last Eigenvector Statistics\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.ylabel('Value')
    plt.grid(True, axis='y')
    
    # we add value labels on bars
    for bar, value in zip(bars, [last_eig_info['barycenter_mean_component'], 
                                 last_eig_info['barycenter_std_component'],
                                 last_eig_info['experiment_variance']]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'eigenvector_analysis_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()
    
    # we create a separate plot for cumulative SVD values for each eigenvector order
    print(f"Creating cumulative SVD plots for N{N}_D{D_IN}_M{M}_L{L}...")
    plt.figure(figsize=(15, 10))
    
    # we plot cumulative variance for each eigenvector order
    plt.subplot(2, 1, 1)
    for k in range(n_vectors):
        cumulative_variance = test_results['eigenvector_tests'][k]['pca_uniformity']['cumulative_variance']
        x_vals = range(0, len(cumulative_variance))  # we start from 0
        plt.plot(x_vals, cumulative_variance, 'o-', linewidth=2, markersize=4, 
                label=f'Eigenvector {k+1}', alpha=0.8)
    
    plt.title(f'Cumulative SVD Values by Eigenvector Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('SVD Component Index (0=start)')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    
    # we add reference lines for uniformity
    max_components = max(len(test_results['eigenvector_tests'][k]['pca_uniformity']['cumulative_variance']) 
                        for k in range(n_vectors))
    uniform_line = np.linspace(0, 1, max_components)
    x_uniform = range(0, max_components)  # we start from 0
    plt.plot(x_uniform, uniform_line, '--', color='black', alpha=0.5, 
             linewidth=2, label='Perfect Uniform (reference)')
    
    # we create a focused plot showing differences more clearly
    plt.subplot(2, 1, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, n_vectors))
    
    for k in range(n_vectors):
        cumulative_variance = test_results['eigenvector_tests'][k]['pca_uniformity']['cumulative_variance']
        x_vals = range(0, len(cumulative_variance))  # we start from 0
        
        # we compute difference from uniform distribution (now starting from 0)
        uniform_ref = np.linspace(0, 1, len(cumulative_variance))
        difference = np.array(cumulative_variance) - uniform_ref
        
        plt.plot(x_vals, difference, 'o-', linewidth=2, markersize=4,
                color=colors[k], label=f'Eigenvector {k+1}', alpha=0.8)
    
    plt.title(f'Deviation from Uniform Distribution\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('SVD Component Index (0=start)')
    plt.ylabel('Deviation from Uniform (positive = more concentrated)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VECTORS, f'cumulative_svd_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()
    
    return {
        'avg_uniformity_score': np.mean([t['pca_uniformity']['uniformity_score'] for t in test_results['eigenvector_tests']]),
        'avg_effective_rank': np.mean([t['pca_uniformity']['effective_rank'] for t in test_results['eigenvector_tests']]),
        'avg_participation_ratio': np.mean([t['pca_uniformity']['participation_ratio'] for t in test_results['eigenvector_tests']]),
        'last_eig_barycenter_mean': test_results['last_eigenvector_analysis']['barycenter_mean_component'],
        'last_eig_barycenter_std': test_results['last_eigenvector_analysis']['barycenter_std_component'],
        'last_eig_experiment_variance': test_results['last_eigenvector_analysis']['experiment_variance'],
        'last_eig_is_constant': test_results['last_eigenvector_analysis']['is_constant_threshold'],
        'N': N,
        'D_IN': D_IN,
        'M': M,
        'L': L
    }

# %%
def create_improved_eigenvector_plots(all_results):
    """we create improved plots closeing trends across configurations"""
    if not all_results:
        return
        
    n_configs = len(all_results)
    
    print("Creating improved eigenvector plots across all configurations...")
    # we create plots closeing variation across all configurations
    plt.figure(figsize=(20, 15))
    
    # we plot PCA uniformity trends
    plt.subplot(3, 2, 1)
    uniformity_values = [r['avg_uniformity_score'] for r in all_results]
    plt.plot(range(n_configs), uniformity_values, 'o-', label='Average Uniformity Score')
    plt.title('PCA Uniformity Score Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Uniformity Score (0=clustered, 1=uniform)')
    plt.grid(True)
    plt.legend()
    
    # we plot effective rank trends  
    plt.subplot(3, 2, 2)
    effective_rank_values = [r['avg_effective_rank'] for r in all_results]
    plt.plot(range(n_configs), effective_rank_values, 'o-', label='Average Effective Rank', color='orange')
    plt.title('Effective Rank Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Effective Rank')
    plt.grid(True)
    plt.legend()
    
    # we plot participation ratio trends
    plt.subplot(3, 2, 3)
    participation_values = [r['avg_participation_ratio'] for r in all_results]
    plt.plot(range(n_configs), participation_values, 'o-', label='Average Participation Ratio', color='green')
    plt.title('Participation Ratio Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Participation Ratio')
    plt.grid(True)
    plt.legend()
    
    # we plot last eigenvector barycenter mean
    plt.subplot(3, 2, 4)
    last_eig_means = [r['last_eig_barycenter_mean'] for r in all_results]
    plt.plot(range(n_configs), last_eig_means, 'o-', label='Last Eigenvector Barycenter Mean', color='red')
    plt.title('Last Eigenvector Barycenter Mean Across Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Mean Value')
    plt.grid(True)
    plt.legend()
    
    # we plot last eigenvector barycenter std
    plt.subplot(3, 2, 5)
    last_eig_stds = [r['last_eig_barycenter_std'] for r in all_results]
    plt.plot(range(n_configs), last_eig_stds, 'o-', label='Last Eigenvector Barycenter Std', color='purple')
    plt.title('Last Eigenvector Barycenter Std Across Configurations')
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
    plt.close()
    print("Improved eigenvector plots completed!")

# %%
# we process all files in the data directory
all_results = []
eigenvalue_means_storage = []  # we store eigenvalue means separately
files = [f for f in os.listdir(os.path.join(PATH_TO_DATA, "vectors")) if f.startswith('ntk_eigenvectors_')]

# we sort files by N
files = sorted(files, key=lambda x: get_config_from_filename(x)[0])

print("Processing all experiment files...")
print("=" * 50)
for file in tqdm(files, desc="Processing experiment files"):
    try:
        # we extract configuration from filename
        N, D_IN, M, L = get_config_from_filename(file)
        
        # we load and analyze data
        eigenvalues_data, eigenvectors_data = load_experiment_data(N, D_IN, M, L)
        
        # we analyze distributions and get eigenvalue means
        eigenvalue_results = plot_eigenvalue_distribution(eigenvalues_data['eigenvalues'], N, D_IN, M, L)
        eigenvector_results = analyze_eigenvector_distribution(eigenvectors_data['eigenvectors'], N, D_IN, M, L)
        
        # we analyze the eigenvalue spacing distribution
        analyze_eigenvalue_spacing(eigenvalues_data['eigenvalues'], N, D_IN, M, L, PATH_TO_PLOTS_SPACING)
        
        # we fit the marchenko-pastur law to the eigenvalue density
        plot_marchenko_pastur_fit(eigenvalues_data, eigenvector_results, PATH_TO_PLOTS_VALUES)
        
        # we test the constant vector hypothesis
        constant_hypothesis_results = test_constant_vector_hypothesis(eigenvectors_data['eigenvectors'], N, D_IN, M, L)
        
        # we combine results
        combined_results = {**eigenvector_results, **eigenvalue_results}
        all_results.append(combined_results)
        eigenvalue_means_storage.append(eigenvalue_results)
        
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("=" * 50)
print("LAST EIGENVECTOR ANALYSIS SUMMARY:")
print("=" * 50)
for result in tqdm(all_results, desc="Printing last eigenvector summary", leave=False):
    config_str = f"N{result['N']}_D{result['D_IN']}_M{result['M']}_L{result['L']}"
    constant_str = "CONSTANT" if result['last_eig_is_constant'] else "VARIABLE"
    print(f"{config_str}: barycenter_mean={result['last_eig_barycenter_mean']:.6f}, barycenter_std={result['last_eig_barycenter_std']:.6f}, exp_var={result['last_eig_experiment_variance']:.6f} [{constant_str}]")

print("\n" + "=" * 50)
print("LAST EIGENVECTOR BARYCENTER (COMPLETE VECTORS):")
print("=" * 50)
for result in tqdm(all_results, desc="Printing last eigenvector barycenters", leave=False):
    config_str = f"N{result['N']}_D{result['D_IN']}_M{result['M']}_L{result['L']}"
    print(f"\n{config_str}:")
    
    # we load the corresponding test results to get the barycenter vector
    test_filename = f'test_results_N{result["N"]}_D{result["D_IN"]}_M{result["M"]}_L{result["L"]}.json'
    try:
        with open(os.path.join(PATH_TO_TESTS, test_filename), 'r') as f:
            test_data = json.load(f)
            barycenter_vector = test_data['last_eigenvector_analysis']['barycenter_vector']
            
        print(f"  Barycenter vector (dimension {len(barycenter_vector)}):")
        # we print vector components in a readable format
        for i, component in enumerate(barycenter_vector):
            print(f"    [{i:2d}]: {component:+.6f}")
            
        # we add some statistics
        print(f"  Statistics:")
        print(f"    Mean of components: {np.mean(barycenter_vector):.6f}")
        print(f"    Std of components:  {np.std(barycenter_vector):.6f}")
        print(f"    Min component:      {np.min(barycenter_vector):+.6f}")
        print(f"    Max component:      {np.max(barycenter_vector):+.6f}")
        print(f"    L2 norm:           {np.linalg.norm(barycenter_vector):.6f}")
            
    except FileNotFoundError:
        print(f"  Error: Test results file not found for {config_str}")
    except Exception as e:
        print(f"  Error loading barycenter for {config_str}: {e}")

print("\n" + "=" * 50)
print("EIGENVALUE MEANS SUMMARY:")
print("=" * 50)
for eig_result in tqdm(eigenvalue_means_storage, desc="Printing eigenvalue summary", leave=False):
    config = eig_result['config']
    config_str = f"N{config['N']}_D{config['D_IN']}_M{config['M']}_L{config['L']}"
    n_eigenvals = len(eig_result['mean_eigenvalues'])
    print(f"{config_str}: {n_eigenvals} eigenvalues (excluding last)")
    for i, mean_val in enumerate(eig_result['mean_eigenvalues'][:5]):  # we close first 5
        print(f"  Eigenvalue {i+1}: {mean_val:.6f}")
    if n_eigenvals > 5:
        print(f"  ... and {n_eigenvals-5} more")

# we create improved plots
create_improved_eigenvector_plots(all_results)

print("Saving results to files...")
# we save results to CSV using native Python
if all_results:
    fieldnames = all_results[0].keys()
    with open(os.path.join(PATH_TO_STATS, 'all_results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in tqdm(all_results, desc="Writing CSV results", leave=False):
            writer.writerow(result)

# we save eigenvalue means separately
print("Saving eigenvalue means...")
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
            'mean_of_barycenter_means': np.mean([r['last_eig_barycenter_mean'] for r in all_results]),
            'mean_of_barycenter_stds': np.mean([r['last_eig_barycenter_std'] for r in all_results]),
            'mean_of_experiment_variances': np.mean([r['last_eig_experiment_variance'] for r in all_results])
        },
        'eigenvalue_analysis': {
            'total_configs_analyzed': len(eigenvalue_means_storage),
            'average_num_eigenvalues': np.mean([len(e['mean_eigenvalues']) for e in eigenvalue_means_storage])
        }
    },
    'results': all_results,
    'eigenvalue_means': eigenvalue_means_storage
}
print("Saving main results JSON...")
with open(os.path.join(PATH_TO_STATS, 'all_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# we plot trends with respect to each parameter using native Python grouping
params = ['L', 'N', 'M', 'D_IN']
metrics = ['avg_uniformity_score', 'avg_effective_rank', 'avg_ks_stat', 'avg_ks_pvalue', 'last_eig_barycenter_mean', 'last_eig_barycenter_std']  # we use PCA metrics

print("Creating parameter trend plots...")
for param in tqdm(params, desc="Creating parameter trend plots"):
    plt.figure(figsize=(20, 8))
    for i, metric in enumerate(tqdm(metrics, desc=f"Plotting {param} trends", leave=False), 1):
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
    plt.close()

print("\nAnalysis complete. Results saved to:")
print(f"- Statistics (CSV & JSON): {PATH_TO_STATS}")
print(f"- Vector plots: {PATH_TO_PLOTS_VECTORS}")
print(f"- Value plots: {PATH_TO_PLOTS_VALUES}")
print(f"- Test results: {PATH_TO_TESTS}")
print(f"- Last eigenvector summary: See JSON metadata section")

# %%
def test_constant_vector_hypothesis(eigenvectors, N, D_IN, M, L):
    """we test if last eigenvector is quasi-constant and others are orthogonal to it"""
    n_experiments, n_vectors, dimension = eigenvectors.shape
    
    # we create the theoretical constant vector (normalized)
    constant_vector = np.ones(dimension) / np.sqrt(dimension)
    
    test_results = {
        'config': {'N': N, 'D_IN': D_IN, 'M': M, 'L': L},
        'constant_vector_tests': [],
        'orthogonality_tests': [],
        'last_eigenvector_analysis': {}
    }
    
    print(f"\nTesting constant vector hypothesis for N{N}_D{D_IN}_M{M}_L{L}:")
    print("=" * 60)
    
    # we test each eigenvector order
    for k in tqdm(range(n_vectors), desc=f"Testing constant hypothesis for N{N}_D{D_IN}_M{M}_L{L}", leave=False):
        vectors_k = eigenvectors[:, k, :]  # we get all experiments for eigenvector k
        
        # we compute barycenter for this eigenvector order
        barycenter_k = np.mean(vectors_k, axis=0)
        
        # we test if barycenter is close to constant vector
        dot_with_constant = np.abs(np.dot(barycenter_k, constant_vector))
        
        # we compute variance of components (low = more constant-like)
        component_variance = np.var(barycenter_k)
        component_range = np.max(barycenter_k) - np.min(barycenter_k)
        
        # we test orthogonality of individual vectors to constant vector
        orthogonality_scores = []
        for exp in range(n_experiments):
            orth_score = np.abs(np.dot(vectors_k[exp], constant_vector))
            orthogonality_scores.append(orth_score)
        
        avg_orthogonality = np.mean(orthogonality_scores)
        std_orthogonality = np.std(orthogonality_scores)
        
        test_results['constant_vector_tests'].append({
            'eigenvector_order': k + 1,
            'dot_with_constant': float(dot_with_constant),
            'component_variance': float(component_variance),
            'component_range': float(component_range),
            'is_constant_like': bool(dot_with_constant > 0.9 and component_variance < 0.01),
            'barycenter': barycenter_k.tolist()
        })
        
        test_results['orthogonality_tests'].append({
            'eigenvector_order': k + 1,
            'avg_orthogonality_score': float(avg_orthogonality),
            'std_orthogonality_score': float(std_orthogonality),
            'is_orthogonal_to_constant': bool(avg_orthogonality < 0.1)
        })
        
        print(f"Eigenvector {k+1:2d}: dot_with_constant={dot_with_constant:.6f}, "
              f"comp_var={component_variance:.6f}, avg_orth={avg_orthogonality:.6f}")
    
    # we analyze the last eigenvector specifically
    last_k = n_vectors - 1
    last_vectors = eigenvectors[:, last_k, :]
    last_barycenter = np.mean(last_vectors, axis=0)
    
    # we test if last eigenvector is the constant direction
    last_dot_constant = np.abs(np.dot(last_barycenter, constant_vector))
    last_is_constant = last_dot_constant > 0.9 and np.var(last_barycenter) < 0.01
    
    test_results['last_eigenvector_analysis'] = {
        'is_constant_direction': bool(last_is_constant),
        'dot_with_constant': float(last_dot_constant),
        'component_variance': float(np.var(last_barycenter)),
        'normalized_barycenter': (last_barycenter / np.linalg.norm(last_barycenter)).tolist(),
        'theoretical_constant': constant_vector.tolist()
    }
    
    # we test uniformity of other eigenvectors on orthogonal space
    print(f"\nAnalyzing distribution on orthogonal space:")
    print("-" * 40)
    
    if last_is_constant:
        print("✅ Last eigenvector is quasi-constant! Testing orthogonal space uniformity...")
        
        # we project all other eigenvectors onto space orthogonal to constant vector
        orthogonal_uniformity = []
        
        for k in range(n_vectors - 1):  # we exclude last eigenvector
            vectors_k = eigenvectors[:, k, :]
            
            # we project onto orthogonal space (remove constant component)
            projected_vectors = []
            for exp in range(n_experiments):
                vec = vectors_k[exp]
                # we remove component in constant direction
                constant_component = np.dot(vec, constant_vector) * constant_vector
                orthogonal_vec = vec - constant_component
                # we renormalize
                if np.linalg.norm(orthogonal_vec) > 1e-10:
                    orthogonal_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)
                projected_vectors.append(orthogonal_vec)
            
            projected_vectors = np.array(projected_vectors)
            
            # we test uniformity of projected vectors
            if len(projected_vectors) > 1:
                pca_results = compute_pca_uniformity(projected_vectors)
                orthogonal_uniformity.append({
                    'eigenvector_order': k + 1,
                    'orthogonal_uniformity_score': pca_results['uniformity_score'],
                    'orthogonal_effective_rank': pca_results['effective_rank']
                })
                
                print(f"Eigenvector {k+1:2d} (orthogonal): uniformity={pca_results['uniformity_score']:.6f}, "
                      f"eff_rank={pca_results['effective_rank']:.2f}")
        
        test_results['orthogonal_space_analysis'] = orthogonal_uniformity
    else:
        print("❌ Last eigenvector is NOT quasi-constant")
        test_results['orthogonal_space_analysis'] = []
    
    # we save results
    test_filename = f'constant_hypothesis_N{N}_D{D_IN}_M{M}_L{L}.json'
    with open(os.path.join(PATH_TO_TESTS, test_filename), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("=" * 60)
    return test_results

def plot_single_config_analysis(test_results, data, vectors_by_order, output_dir):
    """we plot the results of the analysis for a single configuration in a 2x2 grid"""
    N = test_results['N']
    D_IN = test_results['D_IN']
    M = test_results['M']
    L = test_results['L']

    # we extract data for plotting
    eigenvector_indices = [t['index'] + 1 for t in test_results['eigenvector_tests']]  # we start from 1
    uniformity_scores = [t['pca_uniformity']['uniformity_score'] for t in test_results['eigenvector_tests']]
    effective_ranks = [t['pca_uniformity']['effective_rank'] for t in test_results['eigenvector_tests']]
    participation_ratios = [t['pca_uniformity']['participation_ratio'] for t in test_results['eigenvector_tests']]

    plt.figure(figsize=(18, 16))
    plt.suptitle(f'Eigenvector Distribution Analysis for Config N{N}_D{D_IN}_M{M}_L{L}', fontsize=20)

    # we plot PCA/SVD metrics vs eigenvector order
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('PCA/SVD Metrics vs Eigenvector Order')
    ax1.set_xlabel('Eigenvector Order')
    
    color = 'tab:blue'
    ax1.set_ylabel('Rank / Ratio', color=color)
    ax1.plot(eigenvector_indices, effective_ranks, 'o-', color='orange', label='Effective Rank')
    ax1.plot(eigenvector_indices, participation_ratios, 's-', color='green', label='Participation Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax1b = ax1.twinx()
    color = 'tab:red'
    ax1b.set_ylabel('Uniformity Score', color=color)
    ax1b.plot(eigenvector_indices, uniformity_scores, 'd--', color='red', label='Uniformity Score')
    ax1b.tick_params(axis='y', labelcolor=color)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    # we plot 2D PCA of first eigenvectors
    ax2 = plt.subplot(2, 2, 2)
    vectors_k0 = np.array(vectors_by_order[0])
    centered_k0 = vectors_k0 - np.mean(vectors_k0, axis=0)
    u, s, vh = np.linalg.svd(centered_k0, full_matrices=False)
    projection_k0 = u[:, :2] * s[:2]
    ax2.scatter(projection_k0[:, 0], projection_k0[:, 1], alpha=0.6, s=50)
    ax2.set_title('2D PCA of First Eigenvectors (k=1)')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.grid(True)
    ax2.axis('equal')
    
    # we plot last eigenvector barycenter components
    ax3 = plt.subplot(2, 2, 3)
    barycenter_components = test_results['last_eigenvector_analysis']['barycenter_vector']
    d = len(barycenter_components)
    ax3.bar(range(1, d + 1), barycenter_components, label='Barycenter Components', color='purple')
    
    constant_val = 1 / np.sqrt(d)
    ax3.axhline(y=constant_val, color='r', linestyle='--', label=f'Constant value (1/√d) ≈ {constant_val:.3f}')
    
    ax3.set_title(f'Last Eigenvector Barycenter (dim={d})')
    ax3.set_xlabel('Component Index')
    ax3.set_ylabel('Component Value')
    ax3.legend()
    ax3.grid(axis='y')

    # we plot 2D PCA of last eigenvectors
    ax4 = plt.subplot(2, 2, 4)
    last_k_index = max(vectors_by_order.keys())
    vectors_last = np.array(vectors_by_order[last_k_index])
    centered_last = vectors_last - np.mean(vectors_last, axis=0)
    u_last, s_last, vh_last = np.linalg.svd(centered_last, full_matrices=False)
    projection_last = u_last[:, :2] * s_last[:2]
    
    ax4.scatter(projection_last[:, 0], projection_last[:, 1], alpha=0.6, s=50, color='crimson')
    ax4.set_title(f'2D PCA of Last Eigenvectors (k={last_k_index + 1})')
    ax4.set_xlabel('Principal Component 1')
    ax4.set_ylabel('Principal Component 2')
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(output_dir, f'config_N{N}_D{D_IN}_M{M}_L{L}_analysis.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

def plot_cumulative_svd_analysis(test_results, output_dir, N, D_IN, M, L):
    """we plot the cumulative SVD analysis for a single configuration"""
    # ... existing code ...

def plot_marchenko_pastur_fit(data, test_results, output_dir):
    """
    we fit the eigenvalue distribution (excluding the last one) with the Marchenko-Pastur law
    and plot the result.
    """
    N = test_results['N']
    M = test_results['M']
    L = test_results['L']
    D_IN = test_results['D_IN']
    
    # we use eigenvalues from the first experiment as representative
    eigenvalues = np.array(data['eigenvalues'][0])
    
    # we handle the case N > M, where there are N-M zero eigenvalues
    if N > M:
        # we filter out zero eigenvalues for MP fit
        eigs_to_consider = eigenvalues[eigenvalues > 1e-9]
        if len(eigs_to_consider) < 2:
            print(f"Warning: Not enough positive eigenvalues for MP fit for config N{N}_M{M}.")
            return
        eigs_bulk = eigs_to_consider[:-1]
        last_eig = eigs_to_consider[-1]
        gamma = M / N
    else: # N <= M
        eigs_bulk = eigenvalues[:-1]
        last_eig = eigenvalues[-1]
        gamma = N / M

    if len(eigs_bulk) == 0:
        print(f"Warning: Eigenvalue bulk is empty for config N{N}_M{M}.")
        return

    # we fit the Marchenko-Pastur distribution by scaling it to the empirical bulk
    lambda_max_emp = np.max(eigs_bulk)
    # the scaling factor 'c' corresponds to the variance of the matrix elements in the standard MP law
    c = lambda_max_emp / (1 + np.sqrt(gamma))**2 if (1 + np.sqrt(gamma))**2 > 0 else 0
    
    lambda_plus = c * (1 + np.sqrt(gamma))**2
    lambda_minus = c * (1 - np.sqrt(gamma))**2
    
    # we define the Marchenko-Pastur PDF
    def marchenko_pastur_pdf(x, l_plus, l_minus, gamma, c):
        if c == 0 or gamma == 0: return np.zeros_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            # we ensure we are within the support [l_minus, l_plus]
            pdf = np.sqrt(np.maximum(0, l_plus - x) * np.maximum(0, x - l_minus)) / (2 * np.pi * gamma * x * c)
        pdf[np.isnan(pdf)] = 0
        return pdf

    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Marchenko-Pastur Fit for Eigenvalue Spectrum\nConfig N{N}_D{D_IN}_M{M}_L{L}', fontsize=16)
    
    # we plot the empirical histogram
    plt.hist(eigs_bulk, bins=50, density=True, label='Empirical Eigenvalue Distribution (bulk)', alpha=0.7)
    
    # we plot the theoretical MP distribution
    x = np.linspace(lambda_minus, lambda_plus, 400)
    pdf = marchenko_pastur_pdf(x, lambda_plus, lambda_minus, gamma, c)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Marchenko-Pastur Fit (γ={gamma:.2f})')
    
    # we indicate the position of the last eigenvalue
    plt.axvline(x=last_eig, color='g', linestyle='--', linewidth=2, label=f'Last Eigenvalue = {last_eig:.3f}')
    
    # we check the case where M is close to N
    if abs(N - M) / N <= 0.1: # if N and M are within 10% of each other
        plt.text(0.6, 0.8, 'N ≈ M: Last eigenvalue is\nexpected near the bulk edge', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(f'Bulk Support: [{lambda_minus:.3f}, {lambda_plus:.3f}]')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(output_dir, f'config_N{N}_D{D_IN}_M{M}_L{L}_mp_fit.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

def plot_trend_analysis(all_results, output_dir):
    """we plot trends across all configurations"""
    # ... existing code ...
