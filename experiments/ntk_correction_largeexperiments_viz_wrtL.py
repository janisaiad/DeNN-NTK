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
# # Analysis of NTK Correction Term Scaling Laws with respect to L
#
# We analyze how the spectral radius of the NTK correction term scales with respect to network depth (L)

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import linregress

# %%
PATH_TO_DATA = "/home/janis/STG3A/deeperorwider/experiments/data/large_ntk_corrections"
files = os.listdir(PATH_TO_DATA)
data = []

for f in files:
    if f.startswith("ntk_correction_"):
        d = np.load(os.path.join(PATH_TO_DATA, f), allow_pickle=True).item()
        data.append(d)

# %% [markdown]
# ## Scaling Analysis by Configuration
# %%
def plot_config_scaling(data):
    # Create figure with 3 vertically stacked subplots for L, D_IN and N scaling
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18))
    
    # Plot L scaling
    L_groups = {}
    L_slopes = []
    for d in data:
        key = (d['N'], d['D_IN'], d['M'])
        if key not in L_groups:
            L_groups[key] = []
        L_groups[key].append((d['L'], d['mean_spectral_radius']))
    
    # Calculate and store L slopes
    for config, values in L_groups.items():
        if len(values) > 4:
            x = np.array([v[0] for v in sorted(values)])
            y = np.array([v[1] for v in sorted(values)])
            slope, _, r_value, _, _ = linregress(np.log(x), np.log(y))
            L_slopes.append((config, slope, r_value**2, len(x)))
    
    # Plot D_IN scaling
    D_groups = {}
    for d in data:
        key = (d['N'], d['L'], d['M'])
        if key not in D_groups:
            D_groups[key] = []
        D_groups[key].append((d['D_IN'], d['mean_spectral_radius']))
        
    # Plot N scaling
    N_groups = {}
    N_slopes = []
    for d in data:
        key = (d['L'], d['D_IN'], d['M'])
        if key not in N_groups:
            N_groups[key] = []
        N_groups[key].append((d['N'], d['mean_spectral_radius']))

    # Calculate and store N slopes
    for config, values in N_groups.items():
        if len(values) > 4:
            x = np.array([v[0] for v in sorted(values)])
            y = np.array([v[1] for v in sorted(values)])
            slope, _, r_value, _, _ = linregress(np.log(x), np.log(y))
            N_slopes.append((config, slope, r_value**2, len(x)))

    # Plot L scaling
    for config, values in L_groups.items():
        if len(values) > 4:
            x = [v[0] for v in sorted(values)]
            y = [v[1] for v in sorted(values)]
            ax1.loglog(x, y, 'o', alpha=0.3, markersize=4)
            
    ax1.set_title('Spectral Radius vs L (all configs)')
    ax1.set_xlabel('L')
    ax1.set_ylabel('Spectral Radius')
    ax1.grid(True)

    # Plot D_IN scaling
    for config, values in D_groups.items():
        if len(values) > 4:
            x = [v[0] for v in sorted(values)]
            y = [v[1] for v in sorted(values)]
            ax2.loglog(x, y, 'o', alpha=0.3, markersize=4)
            
    ax2.set_title('Spectral Radius vs D_IN (all configs)')
    ax2.set_xlabel('D_IN')
    ax2.set_ylabel('Spectral Radius')
    ax2.grid(True)

    # Plot N scaling
    for config, values in N_groups.items():
        if len(values) > 4:
            x = [v[0] for v in sorted(values)]
            y = [v[1] for v in sorted(values)]
            ax3.loglog(x, y, 'o', alpha=0.3, markersize=4)
            
    ax3.set_title('Spectral Radius vs N (all configs)')
    ax3.set_xlabel('N')
    ax3.set_ylabel('Spectral Radius')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # Print slope tables
    print("\nL Scaling Slopes:")
    print("Config (N, D_IN, M) | Slope | R^2 | Points")
    print("-" * 50)
    for config, slope, r2, points in sorted(L_slopes, key=lambda x: abs(x[1]), reverse=True):
        print(f"N={config[0]}, D={config[1]}, M={config[2]} | {slope:.3f} | {r2:.3f} | {points}")

    print("\nN Scaling Slopes:")
    print("Config (L, D_IN, M) | Slope | R^2 | Points")
    print("-" * 50)
    for config, slope, r2, points in sorted(N_slopes, key=lambda x: abs(x[1]), reverse=True):
        print(f"L={config[0]}, D={config[1]}, M={config[2]} | {slope:.3f} | {r2:.3f} | {points}")

# %%
# Plot scaling with respect to L, D_IN and N
print("Analyzing scaling laws...")
plot_config_scaling(data)
