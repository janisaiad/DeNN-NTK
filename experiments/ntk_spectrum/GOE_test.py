import os
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from collections import defaultdict
import json
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_func
from scipy.stats import norm, arcsine

import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")
PATH_TO_STATS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "stats")
PATH_TO_PLOTS_VECTORS = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "vectors")
PATH_TO_PLOTS_VALUES = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "values")
PATH_TO_PLOTS_SPACING = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "spacing")
PATH_TO_TESTS = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen", "tests")
PATH_TO_PLOTS_GOE = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "goe")

for path in [PATH_TO_STATS, PATH_TO_PLOTS_VECTORS, PATH_TO_PLOTS_VALUES, PATH_TO_TESTS, PATH_TO_PLOTS_SPACING, PATH_TO_PLOTS_GOE]:
    os.makedirs(path, exist_ok=True)

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

def plot_eigenvalue_distribution(eigenvalues, N, D_IN, M, L):
    """we plot histogram of eigenvalues distribution for each k and mean eigenvalues vs order"""
    n_eigenvalues = eigenvalues.shape[1]
    mean_eigenvalues = np.mean(eigenvalues[:, :-1], axis=0)
    eigenvalue_orders = list(range(1, len(mean_eigenvalues) + 1))
    
    plt.figure(figsize=(15, 2*n_eigenvalues + 5))
    
    for k in tqdm(range(n_eigenvalues), desc=f"Plotting eigenvalue distributions for N{N}_D{D_IN}_M{M}_L{L}", leave=False):
        plt.subplot(n_eigenvalues + 1, 2, 2*k + 1)
        plt.hist(eigenvalues[:, k], bins='auto', density=True)
        plt.title(f'Eigenvalue {k+1} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
        
        mean_val = np.mean(eigenvalues[:, k])
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
    
    plt.subplot(n_eigenvalues + 1, 2, 2*n_eigenvalues + 1)
    plt.plot(eigenvalue_orders, mean_eigenvalues, 'o-', linewidth=2, markersize=8, color='blue')
    plt.title(f'Mean Eigenvalues vs Order (Excluding Last)\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvalue Order')
    plt.ylabel('Mean Eigenvalue')
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(n_eigenvalues + 1, 2, 2*n_eigenvalues + 2)
    if len(mean_eigenvalues) > 1:
        eigenvalue_gaps = np.diff(mean_eigenvalues)
        gap_orders = list(range(1, len(eigenvalue_gaps) + 1))
        plt.plot(gap_orders, eigenvalue_gaps, 'o-', linewidth=2, markersize=8, color='orange')
        plt.title(f'Eigenvalue Gaps vs Order\nConfig N{N}_D{D_IN}_M{M}_L{L}')
        plt.xlabel('Gap Order (between k and k+1)')
        plt.ylabel('Eigenvalue Gap')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS_VALUES, f'eigenvalue_dist_N{N}_D{D_IN}_M{M}_L{L}.png'))
    plt.close()

def analyze_eigenvalue_spacing(eigenvalues, N, D_IN, M, L, output_dir):
    """
    i analyze the distribution of the ratio of consecutive eigenvalue spacings
    and compare it to theoretical distributions from random matrix theory (rmt).
    """
    all_spacing_ratios = []
    all_unfolded_spacings = []
    for i in range(eigenvalues.shape[0]):
        eigs = np.sort(eigenvalues[i, :])
        eigs = eigs[eigs > 1e-12] 
        if len(eigs) < 4:
            continue
        
        eigs_bulk = eigs[:-1]
        spacings = np.diff(eigs_bulk)
        spacings = spacings[spacings > 1e-12]
        
        if len(spacings) < 2:
            continue
            
        ratios = spacings[1:] / spacings[:-1]
        all_spacing_ratios.extend(ratios)

        mean_spacing = np.mean(spacings)
        if mean_spacing > 1e-9:
            unfolded_spacings = spacings / mean_spacing
            all_unfolded_spacings.extend(unfolded_spacings)

    if not all_spacing_ratios:
        print(f"Warning: No spacing ratios computed for config N{N}_D{D_IN}_M{M}_L{L}.")
        return

    all_spacing_ratios = np.array(all_spacing_ratios)
    mean_min_r = np.mean(np.minimum(all_spacing_ratios, 1/all_spacing_ratios))

    plt.figure(figsize=(12, 8))
    plt.hist(all_spacing_ratios, bins="auto", density=True, label='Empirical Ratios', range=(0, 10), color='skyblue')
    
    r = np.linspace(0, 10, 400)
    p_poisson = 1 / (1 + r)**2
    p_goe = (27 / 8) * (r + r**2) / (1 + r + r**2)**2.5
    
    plt.plot(r, p_poisson, 'g--', linewidth=2, label='Poisson (Uncorrelated)')
    plt.plot(r, p_goe, 'r-', linewidth=2, label='GOE (Wigner-Dyson)')
    
    plt.title(f'² Ratio Distribution\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Ratio of Consecutive Spacings (r)')
    plt.ylabel('Probability Density P(r)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    stats_text = (
        f"Empirical <min(r, 1/r)> = {mean_min_r:.4f}\n\n"
        f"Theoretical <min(r, 1/r)>:\n"
        f"  Poisson: {np.log(2):.4f}\n"
        f"  GOE: {0.535:.4f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plot_filename = os.path.join(output_dir, f'consecutive_spacing_ratio_dist_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close()
    
    if all_unfolded_spacings:
        all_unfolded_spacings = np.array(all_unfolded_spacings)
        plt.figure(figsize=(12, 8))
        plt.hist(all_unfolded_spacings, bins="auto", density=True, label='Empirical Unfolded Spacings', range=(0, 4), color='skyblue')

        s = np.linspace(0, 4, 400)
        p_poisson_s = np.exp(-s)
        p_goe_s = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

        plt.plot(s, p_poisson_s, 'g--', linewidth=2, label='Poisson (q=0)')
        plt.plot(s, p_goe_s, 'r-', linewidth=2, label='GOE (q=1)')
        
        plt.title(f'Unfolded Eigenvalue Spacing Distribution\nConfig N{N}_D{D_IN}_M{M}_L{L}')
        plt.xlabel('Unfolded Spacing (s)')
        plt.ylabel('Probability Density P(s)')
        plt.legend()
        plt.grid(True, alpha=0.5)

        brody_plot_filename = os.path.join(output_dir, f'brody_fit_N{N}_D{D_IN}_M{M}_L{L}.png')
        plt.savefig(brody_plot_filename, dpi=120, bbox_inches='tight')
        plt.close()

def compute_pca_uniformity(vectors):
    """we use PCA/SVD to test uniformity - uniform distribution should have equal singular values"""
    _, dimension = vectors.shape
    if dimension == 0:
        return {'uniformity_score': 0, 'effective_rank': 0}
    centered_vectors = vectors - np.mean(vectors, axis=0)
    _, singular_values, _ = np.linalg.svd(centered_vectors, full_matrices=False)
    
    if len(singular_values) == 0 or np.sum(singular_values**2) == 0:
        return {'uniformity_score': 0, 'effective_rank': 0}

    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    eps = 1e-10
    svd_entropy = -np.sum(explained_variance * np.log(explained_variance + eps))
    max_entropy = np.log(dimension)
    normalized_entropy = svd_entropy / max_entropy if max_entropy > 0 else 0
    effective_rank = np.exp(svd_entropy)
    
    return {'uniformity_score': float(normalized_entropy), 'effective_rank': float(effective_rank)}

def test_constant_vector_hypothesis(eigenvectors, N, D_IN, M, L):
    """we test if last eigenvector is quasi-constant and others are orthogonal to it"""
    n_experiments, n_vectors, dimension = eigenvectors.shape
    if dimension == 0: return
    
    constant_vector = np.ones(dimension) / np.sqrt(dimension)
    
    print(f"\nTesting constant vector hypothesis for N{N}_D{D_IN}_M{M}_L{L}:")
    
    last_vectors = eigenvectors[:, -1, :]
    last_barycenter = np.mean(last_vectors, axis=0)
    last_dot_constant = np.abs(np.dot(last_barycenter, constant_vector))
    last_is_constant = last_dot_constant > 0.95 and np.var(last_barycenter) < 0.01
    
    if last_is_constant:
        print("✅ Last eigenvector is quasi-constant. Testing orthogonal space uniformity...")
        for k in range(n_vectors - 1):
            vectors_k = eigenvectors[:, k, :]
            
            projected_vectors = []
            for exp in range(n_experiments):
                vec = vectors_k[exp]
                constant_component = np.dot(vec, constant_vector) * constant_vector
                orthogonal_vec = vec - constant_component
                norm_val = np.linalg.norm(orthogonal_vec)
                if norm_val > 1e-10:
                    orthogonal_vec /= norm_val
                projected_vectors.append(orthogonal_vec)
            
            projected_vectors = np.array(projected_vectors)
            
            if len(projected_vectors) > 1:
                pca_results = compute_pca_uniformity(projected_vectors)
                print(f"  Eigenvector {k+1:2d} (ortho): uniformity={pca_results['uniformity_score']:.4f}")
    else:
        print(f"❌ Last eigenvector is NOT quasi-constant (dot product: {last_dot_constant:.4f})")

def fit_arcsine_to_eigenvalues(eigenvectors, N, D_IN, M, L, output_dir):
    """
    i test if the eigenvalue density of the change-of-basis matrix fits an arcsine distribution.
    """
    n_experiments, _, n_dim = eigenvectors.shape
    if n_experiments < 2 or n_dim < N: return

    basis_ortho = eigenvectors[0, :, :N-1]
    all_symmetrized_eigenvalues = []
    for i in range(1, n_experiments):
        vectors_i_ortho = eigenvectors[i, :, :N-1]
        c_i = basis_ortho.T @ vectors_i_ortho
        s_i = (c_i + c_i.T) / 2
        eigenvalues_si = np.linalg.eigh(s_i)[0]
        all_symmetrized_eigenvalues.append(eigenvalues_si)

    if not all_symmetrized_eigenvalues: return

    flat_eigenvalues = np.concatenate(all_symmetrized_eigenvalues)
    if len(flat_eigenvalues) == 0: return

    # we fit the arcsine distribution by scaling to the data's range
    y_min, y_max = np.min(flat_eigenvalues), np.max(flat_eigenvalues)
    if y_max - y_min < 1e-9: return # we avoid division by zero if all values are same

    x = np.linspace(y_min, y_max, 400)
    arcsine_pdf = arcsine.pdf(x, loc=y_min, scale=y_max - y_min)

    plt.figure(figsize=(10, 8))
    plt.hist(flat_eigenvalues, bins="auto", density=True, label='Empirical Eigenvalue Density', alpha=0.7, color='skyblue')
    plt.plot(x, arcsine_pdf, 'r-', linewidth=2, label=f'Arcsine Fit')
    plt.title(f'Subspace Matrix Eigenvalue Density vs Arcsine Law\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(output_dir, f'arcsine_fit_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120)
    plt.close()
    
    print(f"Arcsine fit for N{N}_D{D_IN}_M{M}_L{L} complete.")

def test_goe_trace_density(eigenvectors, N, D_IN, M, L, output_dir):
    """
    i test if the trace of the change-of-basis matrices follows a gaussian distribution, as predicted by goe.
    """
    n_experiments, _, n_dim = eigenvectors.shape
    if n_experiments < 2 or n_dim < N: return

    basis_ortho = eigenvectors[0, :, :N-1]
    traces = []
    
    for i in range(1, n_experiments):
        vectors_i_ortho = eigenvectors[i, :, :N-1]
        c_i = basis_ortho.T @ vectors_i_ortho
        s_i = (c_i + c_i.T) / 2
        traces.append(np.trace(s_i))

    if not traces: return
    
    traces = np.array(traces)
    empirical_mean, empirical_std = np.mean(traces), np.std(traces)
    
    if empirical_std < 1e-9: return # not enough variance to plot

    plt.figure(figsize=(10, 8))
    plt.hist(traces, bins='auto', density=True, label='Empirical Trace Density', alpha=0.7, color='skyblue')

    # we plot theoretical gaussian centered at the empirical mean
    x = np.linspace(empirical_mean - 3*empirical_std, empirical_mean + 3*empirical_std, 200)
    theoretical_pdf = norm.pdf(x, empirical_mean, empirical_std)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label=f'Gaussian Fit (μ={empirical_mean:.2f}, σ={empirical_std:.2f})')
    
    plt.title(f'Trace Density of Subspace Matrices vs Gaussian Fit\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Trace Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plot_filename = os.path.join(output_dir, f'goe_trace_density_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120)
    plt.close()
    
    print(f"GOE trace density test for N{N}_D{D_IN}_M{M}_L{L} complete.")

def main():
    """main processing loop."""
    files = [f for f in os.listdir(os.path.join(PATH_TO_DATA, "vectors")) if f.startswith('ntk_eigenvectors_')]
    files = sorted(files, key=lambda x: (-get_config_from_filename(x)[0], get_config_from_filename(x)[1], get_config_from_filename(x)[2], get_config_from_filename(x)[3]))

    print("Processing all experiment files...")
    print("=" * 50)
    for file in tqdm(files, desc="Processing experiment files"):
        try:
            N, D_IN, M, L = get_config_from_filename(file)
            eigenvectors_data = load_experiment_data(N, D_IN, M, L)
            if eigenvectors_data is None:
                print(f"Could not find data for {file}, skipping.")
                continue

            eigenvectors = eigenvectors_data['eigenvectors']

            test_constant_vector_hypothesis(eigenvectors, N, D_IN, M, L)
            fit_arcsine_to_eigenvalues(eigenvectors, N, D_IN, M, L, PATH_TO_PLOTS_GOE)
            test_goe_trace_density(eigenvectors, N, D_IN, M, L, PATH_TO_PLOTS_GOE)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAnalysis complete. Results saved to:")
    print(f"- GOE plots: {PATH_TO_PLOTS_GOE}")

if __name__ == "__main__":
    main() 