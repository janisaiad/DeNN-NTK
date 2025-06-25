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
# # Analysis of NTK Correction Term Scaling Laws
#
# We analyze how the spectral radius of the NTK correction term scales wrt:
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
from sklearn.linear_model import LinearRegression

# %%
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "large_ntk_corrections")
files = os.listdir(PATH_TO_DATA)
data = []

for f in files:
    if f.startswith("ntk_correction_"):
        d = np.load(os.path.join(PATH_TO_DATA, f), allow_pickle=True).item()
        data.append(d)

# %% [markdown]
# ## Scaling Analysis by Configuration
# %%
def plot_config_scaling(data, vary_param, fixed_params):
    
    unique_values = {p: sorted(list(set(d[p] for d in data))) for p in fixed_params} # i get unique values for each fixed parameter
    
    groups = {} # i group data by fixed parameter combinations
    slopes = []
    slopes_file = f"ntk_slopes_{vary_param}.txt"
    with open(slopes_file, 'w') as f:
        f.write(f"Slopes for {vary_param} scaling:\n")
        f.write("Configuration | Slope | R^2 | Points\n")
        f.write("-" * 50 + "\n")
    
    for d in data:
        key = tuple(d[p] for p in fixed_params)
        if key not in groups:
            groups[key] = []
        groups[key].append((d[vary_param], d['mean_spectral_radius'], d['std_spectral_radius']))

    filtered_groups = {k:v for k,v in groups.items() if len(v) > 3} # i keep only groups with >3 points
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)

    if not sorted_groups:
        print(f"No configurations found with more than 3 points for {vary_param}")
        return

    # i create a separate figure for each plot
    for idx, (config, values) in enumerate(sorted_groups[::5]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_values = sorted(values, key=lambda x: x[0])
        x = np.array([v[0] for v in sorted_values])
        y = np.array([v[1] for v in sorted_values])
        yerr = [v[2] for v in sorted_values]
        
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, markersize=8)
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
        slopes.append((config, slope, r_value**2, len(x)))
        x_line = np.array(sorted(x))
        ax.plot(x_line, np.exp(intercept) * x_line**slope, '--',
                label=f'slope={slope:.2f}')
        
        config_str = ", ".join([f"{p}={v}" for p,v in zip(fixed_params, config)]) # i format config string
        with open(slopes_file, 'a') as f:
            f.write(f"{config_str} | {slope:.3f} | {r_value**2:.3f} | {len(x)}\n")
        
        ax.set_title(f'Spectral Radius vs {vary_param}\n{config_str}')
        ax.set_xlabel(vary_param)
        ax.set_ylabel('Spectral Radius')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    return slopes

# %%
print("Analyzing depth (L) scaling for each configuration...") # analyze L scaling
L_slopes = plot_config_scaling(data, 'L', ['N', 'D_IN', 'M'])

# %%
print("Analyzing input dimension (D) scaling for each configuration...") # analyze D scaling
D_slopes = plot_config_scaling(data, 'D_IN', ['N', 'L', 'M'])

# %%
print("Analyzing number of samples (N) scaling for each configuration...") # analyze N scaling
N_slopes = plot_config_scaling(data, 'N', ['D_IN', 'L', 'M'])

# %%
print("Analyzing width (M) scaling for each configuration...") # analyze M scaling
M_slopes = plot_config_scaling(data, 'M', ['N', 'D_IN', 'L'])

# %%
print("Plotting all points vs L...") # plot L scaling overview
plt.figure(figsize=(10, 6))

configs = [(d['N'], d['D_IN'], d['M']) for d in data] # get unique configurations
unique_configs = list(set(configs))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_configs)))
config_to_color = dict(zip(unique_configs, colors))

for config in unique_configs:
    mask = [(d['N'], d['D_IN'], d['M']) == config for d in data]
    L_values_config = [d['L'] for d, m in zip(data, mask) if m]
    spectral_radii_config = [d['mean_spectral_radius'] for d, m in zip(data, mask) if m]
    plt.scatter(L_values_config, spectral_radii_config, 
               color=config_to_color[config], alpha=0.5,
               label=f'N={config[0]}, D={config[1]}, M={config[2]}')

plt.xlabel('L')
plt.ylabel('Spectral Radius')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.title('All Spectral Radii vs Network Depth (L)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
print("Plotting all points vs D_IN...") # plot D_IN scaling overview
plt.figure(figsize=(10, 6))

for config in unique_configs:
    mask = [(d['N'], d['D_IN'], d['M']) == config for d in data]
    D_values_config = [d['D_IN'] for d, m in zip(data, mask) if m]
    spectral_radii_config = [d['mean_spectral_radius'] for d, m in zip(data, mask) if m]
    plt.scatter(D_values_config, spectral_radii_config, 
               color=config_to_color[config], alpha=0.5,
               label=f'N={config[0]}, D={config[1]}, M={config[2]}')

plt.xlabel('D_IN')
plt.ylabel('Spectral Radius')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.title('All Spectral Radii vs Input Dimension (D_IN)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
print("Plotting all points vs N...") # plot N scaling overview
plt.figure(figsize=(10, 6))

for config in unique_configs:
    mask = [(d['N'], d['D_IN'], d['M']) == config for d in data]
    N_values_config = [d['N'] for d, m in zip(data, mask) if m]
    spectral_radii_config = [d['mean_spectral_radius'] for d, m in zip(data, mask) if m]
    plt.scatter(N_values_config, spectral_radii_config, 
               color=config_to_color[config], alpha=0.5,
               label=f'N={config[0]}, D={config[1]}, M={config[2]}')

plt.xlabel('N')
plt.ylabel('Spectral Radius')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.title('All Spectral Radii vs Number of Samples (N)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
print("Plotting all points vs M...") # plot M scaling overview
plt.figure(figsize=(10, 6))

for config in unique_configs:
    mask = [(d['N'], d['D_IN'], d['M']) == config for d in data]
    M_values_config = [d['M'] for d, m in zip(data, mask) if m]
    spectral_radii_config = [d['mean_spectral_radius'] for d, m in zip(data, mask) if m]
    plt.scatter(M_values_config, spectral_radii_config, 
               color=config_to_color[config], alpha=0.5,
               label=f'N={config[0]}, D={config[1]}, M={config[2]}')

plt.xlabel('M')
plt.ylabel('Spectral Radius')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.title('All Spectral Radii vs Network Width (M)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
print("\nMultivariate Linear Regression Analysis in Log Space:") # perform multivariate regression
X = np.array([[np.log(d['L']), np.log(d['D_IN']), np.log(d['N']), np.log(d['M'])] for d in data])
y = np.array([np.log(d['mean_spectral_radius']) for d in data])

reg = LinearRegression().fit(X, y)

print("\nRegression coefficients:") # print regression results
print(f"L coefficient: {reg.coef_[0]:.3f}")
print(f"D_IN coefficient: {reg.coef_[1]:.3f}")
print(f"N coefficient: {reg.coef_[2]:.3f}")
print(f"M coefficient: {reg.coef_[3]:.3f}")
print(f"Intercept: {reg.intercept_:.3f}")
print(f"R² score: {reg.score(X, y):.3f}")

print("\nThis means the spectral radius scales approximately as:") # show scaling relationship
print(f"spectral_radius ∝ L^{reg.coef_[0]:.3f} * D_IN^{reg.coef_[1]:.3f} * N^{reg.coef_[2]:.3f} * M^{reg.coef_[3]:.3f}")

# %%
