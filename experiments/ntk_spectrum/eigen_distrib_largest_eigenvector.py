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

# %%
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")
PATH_TO_PLOTS = os.path.join(PROJECT_ROOT, "experiments", "plots", "eigen", "largest")
# we create necessary directories
os.makedirs(PATH_TO_PLOTS, exist_ok=True)

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

def plot_largest_eigenvector_scalar_product_distribution(eigenvectors, N, D_IN, M, L, output_dir):
    """
    we analyze the distribution of the largest eigenvectors.
    for each configuration, we compute the mean of the largest eigenvectors (barycenter)
    and then plot the distribution of the scalar products of each largest eigenvector
    with this barycenter.
    """
    # we assume the largest eigenvector is at index 0
    # eigenvectors shape: (n_experiments, n_vectors, dimension)
    for i in range(eigenvectors.shape[0]):
        eigenvectors[i,:,:] = eigenvectors[i,:,:].T
    
    try:
        largest_eigenvectors = eigenvectors[:, -1, :] # shape (n_experiments, dimension)
        print(largest_eigenvectors)
    except IndexError:
        print(f"Warning: Could not extract largest eigenvector for config N{N}_D{D_IN}_M{M}_L{L}. Eigenvector array might be empty or have wrong dimensions.")
        return

    # we compute the barycenter (mean eigenvector)
    barycenter = np.mean(largest_eigenvectors, axis=0)

    # we normalize the barycenter to have unit norm for consistent scalar products
    barycenter_norm = np.linalg.norm(barycenter)
    if barycenter_norm < 1e-10:
        print(f"Warning: Barycenter norm is close to zero for config N{N}_D{D_IN}_M{M}_L{L}. Skipping plot.")
        return
    
    normalized_barycenter = barycenter / barycenter_norm

    # we compute the scalar product of each largest eigenvector with the normalized barycenter
    # eigenvectors are assumed to be unit vectors. The scalar product is the cosine similarity.
    scalar_products = np.dot(largest_eigenvectors, normalized_barycenter)

    # we plot the distribution of these scalar products
    plt.figure(figsize=(10, 6))
    plt.hist(scalar_products, bins='auto', density=True, alpha=0.7, label='Scalar Product Distribution')

    # we add mean and std dev to the plot
    mean_sp = np.mean(scalar_products)
    std_sp = np.std(scalar_products)
    plt.axvline(mean_sp, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_sp:.4f}')
    
    plt.title(f'Distribution of Largest Eigenvector Scalar Products with Barycenter\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Scalar Product (Cosine Similarity)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # we add text box with statistics
    stats_text = f'Std Dev: {std_sp:.4f}\nMin: {np.min(scalar_products):.4f}\nMax: {np.max(scalar_products):.4f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # we save the plot
    plot_filename = os.path.join(output_dir, f'largest_eigenvector_scalar_prod_dist_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120)
    plt.show()
    plt.close()

def plot_largest_eigenvector_coordinate_distributions(eigenvectors, N, D_IN, M, L, output_dir):
    """
    we plot the distribution of each coordinate/component of the largest eigenvectors.
    for each coordinate i, we plot the histogram of values across all experiments.
    """
    # we assume the largest eigenvector is at index -1 (last one)
    # eigenvectors shape: (n_experiments, n_vectors, dimension)
    try:
        largest_eigenvectors = eigenvectors[:, -1, :] # we get shape (n_experiments, dimension)
    except IndexError:
        print(f"Warning: Could not extract largest eigenvector for config N{N}_D{D_IN}_M{M}_L{L}. Eigenvector array might be empty or have wrong dimensions.")
        return

    n_experiments, dimension = largest_eigenvectors.shape
    
    # we compute number of rows and columns for subplots
    n_cols = min(4, dimension)  # we limit to 4 columns maximum
    n_rows = (dimension + n_cols - 1) // n_cols  # we compute rows needed
    
    # we create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig.suptitle(f'Distribution of Each Coordinate - Largest Eigenvector\nConfig N{N}_D{D_IN}_M{M}_L{L}', fontsize=16)
    
    # we flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # we plot distribution for each coordinate
    for i in range(dimension):
        coordinate_values = largest_eigenvectors[:, i]  # we get values for coordinate i across all experiments
        
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # we plot histogram
        ax.hist(coordinate_values, bins='auto', density=True, alpha=0.7, color=f'C{i%10}')
        
        # we add statistics
        mean_val = np.mean(coordinate_values)
        std_val = np.std(coordinate_values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_title(f'Coordinate {i+1}\nMean: {mean_val:.4f}, Std: {std_val:.4f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # we add text box with additional statistics
        stats_text = f'Min: {np.min(coordinate_values):.4f}\nMax: {np.max(coordinate_values):.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # we hide unused subplots
    for i in range(dimension, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # we save the plot
    plot_filename = os.path.join(output_dir, f'largest_eigenvector_coordinates_dist_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # we also create a summary statistics plot
    plt.figure(figsize=(12, 8))
    
    # we compute statistics for all coordinates
    coordinate_means = [np.mean(largest_eigenvectors[:, i]) for i in range(dimension)]
    coordinate_stds = [np.std(largest_eigenvectors[:, i]) for i in range(dimension)]
    coordinate_mins = [np.min(largest_eigenvectors[:, i]) for i in range(dimension)]
    coordinate_maxs = [np.max(largest_eigenvectors[:, i]) for i in range(dimension)]
    
    coordinate_indices = list(range(1, dimension + 1))  # we start from 1
    
    plt.subplot(2, 2, 1)
    plt.plot(coordinate_indices, coordinate_means, 'o-', linewidth=2, markersize=6)
    plt.title(f'Mean Value per Coordinate\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Mean Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(coordinate_indices, coordinate_stds, 'o-', color='orange', linewidth=2, markersize=6)
    plt.title(f'Standard Deviation per Coordinate\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(coordinate_indices, coordinate_mins, 'o-', color='green', linewidth=2, markersize=6, label='Min')
    plt.plot(coordinate_indices, coordinate_maxs, 's-', color='red', linewidth=2, markersize=6, label='Max')
    plt.title(f'Min/Max Values per Coordinate\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    coordinate_ranges = [coordinate_maxs[i] - coordinate_mins[i] for i in range(dimension)]
    plt.plot(coordinate_indices, coordinate_ranges, 'o-', color='purple', linewidth=2, markersize=6)
    plt.title(f'Range (Max-Min) per Coordinate\nConfig N{N}_D{D_IN}_M{M}_L{L}')
    plt.xlabel('Coordinate Index')
    plt.ylabel('Range')
    plt.grid(True)
    
    plt.tight_layout()
    
    # we save the summary plot
    summary_filename = os.path.join(output_dir, f'largest_eigenvector_coordinates_summary_N{N}_D{D_IN}_M{M}_L{L}.png')
    plt.savefig(summary_filename, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Coordinate analysis completed for N{N}_D{D_IN}_M{M}_L{L}")
    print(f"  Dimension: {dimension}")
    print(f"  Overall mean of means: {np.mean(coordinate_means):.6f}")
    print(f"  Overall std of means: {np.std(coordinate_means):.6f}")
    print(f"  Average std across coordinates: {np.mean(coordinate_stds):.6f}")

# %%
if __name__ == "__main__":
    # we process all files in the vectors directory
    files = [f for f in os.listdir(os.path.join(PATH_TO_DATA, "vectors")) if f.startswith('ntk_eigenvectors_')]
    
    # we sort files by N
    files = sorted(files, key=lambda x: get_config_from_filename(x)[0])
    
    files = files[:1]
    print("Processing all experiment files for largest eigenvector analysis...")
    print("=" * 60)
    
    for file in tqdm(files, desc="Processing experiment files"):
        try:
            # we extract configuration from filename
            N, D_IN, M, L = get_config_from_filename(file)
            
            # we load eigenvectors data
            _, eigenvectors_data = load_experiment_data(N, D_IN, M, L)
            
            # we plot the largest eigenvector scalar product distribution
            plot_largest_eigenvector_scalar_product_distribution(
                eigenvectors_data['eigenvectors'], N, D_IN, M, L, PATH_TO_PLOTS
            )
            
            # we plot the largest eigenvector coordinate distributions
            plot_largest_eigenvector_coordinate_distributions(
                eigenvectors_data['eigenvectors'], N, D_IN, M, L, PATH_TO_PLOTS
            )
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print("Largest eigenvector analysis complete!")

# %%
