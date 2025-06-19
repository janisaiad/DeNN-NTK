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
# # Analysis of K3 Tensor Scaling Laws
#
# We analyze how the largest eigenvalue and infinity norm of the K3 tensor scale with respect to:
# - Network depth (L)
# - Network width (M) 
# - Input dimension (D)
# - Number of samples (N)

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import linregress

# %%
PATH_TO_DATA = "/home/janis/STG3A/deeperorwider/experiments/data/large"
files = os.listdir(PATH_TO_DATA)
data = []

for f in files:
    if f.startswith("k3_analysis_"):
        d = np.load(os.path.join(PATH_TO_DATA, f), allow_pickle=True).item()
        data.append(d)

# %% [markdown]
# ## Scaling Analysis by Configuration
# %%
def plot_config_scaling(data, vary_param, fixed_params, metrics=['inf_norm', 'max_eigenvalue']):
    # Get unique values for each fixed parameter
    unique_values = {p: sorted(list(set(d[p] for d in data))) for p in fixed_params}
    
    # Group data by fixed parameter combinations
    groups = {}
    for d in data:
        key = tuple(d[p] for p in fixed_params)
        if key not in groups:
            groups[key] = []
        max_eig = np.max(np.abs(d['mean_eigenvalues']))
        groups[key].append((d[vary_param], d['inf_norm'], max_eig))

    # Sort groups by number of points
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)  # we sort by number of points in descending order

    # Create figure with subplots for each configuration
    n_configs = len(groups)
    fig, axes = plt.subplots(n_configs, 2, figsize=(15, 4*n_configs))
    
    for idx, (config, values) in enumerate(sorted_groups[::5]):  # we use sorted_groups instead of sorted(groups.items())
        # Plot infinity norm
        ax = axes[idx, 0]
        sorted_values = sorted(values, key=lambda x: x[0])
        x = [v[0] for v in sorted_values]
        y = [v[1] for v in sorted_values]
        
        ax.scatter(x, y, s=100)
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
            x_line = np.array(sorted(x))
            ax.plot(x_line, np.exp(intercept) * x_line**slope, '--',
                    label=f'slope={slope:.2f}')
        
        config_str = ", ".join([f"{p}={v}" for p,v in zip(fixed_params, config)])
        ax.set_title(f'Infinity Norm vs {vary_param}\n{config_str}')
        ax.set_xlabel(vary_param)
        ax.set_ylabel('Infinity Norm')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

        # Plot max eigenvalue
        ax = axes[idx, 1]
        y = [v[2] for v in sorted_values]
        
        ax.scatter(x, y, s=100)
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
            x_line = np.array(sorted(x))
            ax.plot(x_line, np.exp(intercept) * x_line**slope, '--',
                    label=f'slope={slope:.2f}')
        
        ax.set_title(f'Max Eigenvalue vs {vary_param}\n{config_str}')
        ax.set_xlabel(vary_param)
        ax.set_ylabel('Max Eigenvalue')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# %%
# Plot scaling with respect to L for each (N,D,M) configuration
print("Analyzing depth (L) scaling for each configuration...")
plot_config_scaling(data, 'L', ['N', 'D_IN', 'M'])

# %%
# Plot scaling with respect to D for each (N,L,M) configuration  
print("Analyzing input dimension (D) scaling for each configuration...")
plot_config_scaling(data, 'D_IN', ['N', 'L', 'M'])

# %%
