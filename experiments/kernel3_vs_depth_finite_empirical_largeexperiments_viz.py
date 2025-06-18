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
# ## Scaling Laws Analysis

# %%
def analyze_parameter_scaling(data, param_name, plot_size=(20, 15)):  # we increase figure size
    other_params = ['N', 'D_IN', 'M', 'L']
    other_params.remove(param_name)
    
    groups = {}
    for d in data:
        key = tuple(d[p] for p in other_params)
        if key not in groups:
            groups[key] = []
        max_eigenvalue = np.max(np.abs(d['mean_eigenvalues']))  # we get largest eigenvalue
        groups[key].append((d[param_name], d['inf_norm'], max_eigenvalue))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=plot_size)
    
    # Plot 1: Infinity Norm vs Parameter (Linear Scale)
    ax = axes[0,0]
    for key, values in groups.items():
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        ax.scatter(x, y, label=str(key), s=100)  # we use scatter plot with larger markers
    ax.set_xlabel(param_name)
    ax.set_ylabel('Infinity Norm')
    ax.set_title(f'K3 Infinity Norm vs {param_name} (Linear)')
    ax.grid(True)
    
    # Plot 2: Infinity Norm vs Parameter (Log Scale)
    ax = axes[0,1]
    for key, values in groups.items():
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        ax.scatter(x, y, label=str(key), s=100)  # we use scatter plot with larger markers
        
        if len(x) > 1:  # we fit power law if enough points
            slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
            ax.plot(x, np.exp(intercept) * np.array(x)**slope, '--',
                    label=f'slope={slope:.2f}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Infinity Norm')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'K3 Infinity Norm vs {param_name} (Log)')
    ax.grid(True)
    
    # Plot 3: Largest Eigenvalue vs Parameter (Linear Scale)
    ax = axes[1,0]
    for key, values in groups.items():
        x = [v[0] for v in values]
        y = [v[2] for v in values]
        ax.scatter(x, y, label=str(key), s=100)  # we use scatter plot with larger markers
    ax.set_xlabel(param_name)
    ax.set_ylabel('Largest Eigenvalue')
    ax.set_title(f'K3 Largest Eigenvalue vs {param_name} (Linear)')
    ax.grid(True)
    
    # Plot 4: Largest Eigenvalue vs Parameter (Log Scale)
    ax = axes[1,1]
    for key, values in groups.items():
        x = [v[0] for v in values]
        y = [v[2] for v in values]
        ax.scatter(x, y, label=str(key), s=100)  # we use scatter plot with larger markers
        
        if len(x) > 1:  # we fit power law if enough points
            slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
            ax.plot(x, np.exp(intercept) * np.array(x)**slope, '--',
                    label=f'slope={slope:.2f}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Largest Eigenvalue')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'K3 Largest Eigenvalue vs {param_name} (Log)')
    ax.grid(True)
    
    # Add legend outside plots
    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.show()

# %%
# Analyze depth scaling
print("Analyzing depth (L) scaling relationships...")
analyze_parameter_scaling(data, 'L')

# %%
# Analyze width scaling
print("Analyzing width (M) scaling relationships...")
analyze_parameter_scaling(data, 'M')

# %%
# Analyze input dimension scaling
print("Analyzing input dimension (D_IN) scaling relationships...")
analyze_parameter_scaling(data, 'D_IN')

# %%
# Analyze sample size scaling
print("Analyzing sample size (N) scaling relationships...")
analyze_parameter_scaling(data, 'N')
