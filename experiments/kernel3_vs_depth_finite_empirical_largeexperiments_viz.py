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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
def plot_config_scaling(data, vary_param, fixed_params):
    
    unique_values = {p: sorted(list(set(d[p] for d in data))) for p in fixed_params} # i get unique values for each fixed parameter
    
    groups = {} # i group data by fixed parameter combinations
    slopes = []
    slopes_file = f"kernel3_slopes_{vary_param}.txt"
    with open(slopes_file, 'w') as f:
        f.write(f"Slopes for {vary_param} scaling:\n")
        f.write("Configuration | Slope (Inf Norm) | Slope (Max Eig) | R^2 | Points\n")
        f.write("-" * 70 + "\n")
    
    for d in data:
        key = tuple(d[p] for p in fixed_params)
        if key not in groups:
            groups[key] = []
        max_eigenvalue = np.max(np.abs(d['mean_eigenvalues']))
        max_eigenvalue_std = d['std_eigenvalues'][np.argmax(np.abs(d['mean_eigenvalues']))]
        groups[key].append((d[vary_param], d['inf_norm'], max_eigenvalue, max_eigenvalue_std))

    filtered_groups = {k:v for k,v in groups.items() if len(v) > 3} # i keep only groups with >3 points
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)

    if not sorted_groups:
        print(f"No configurations found with more than 3 points for {vary_param}")
        return

    # i create figures with subplots side by side
    for idx, (config, values) in enumerate(sorted_groups[::5]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        sorted_values = sorted(values, key=lambda x: x[0])
        x = np.array([v[0] for v in sorted_values])
        y_inf = np.array([v[1] for v in sorted_values])
        y_eig = np.array([v[2] for v in sorted_values])
        y_eig_std = np.array([v[3] for v in sorted_values])
        
        # Infinity Norm Plot
        ax1.scatter(x, y_inf, marker='o', s=80)
        slope_inf, intercept_inf, r_value_inf, _, _ = linregress(np.log(x), np.log(y_inf))
        x_line = np.array(sorted(x))
        ax1.plot(x_line, np.exp(intercept_inf) * x_line**slope_inf, '--',
                label=f'slope={slope_inf:.2f}')
        
        config_str = ", ".join([f"{p}={v}" for p,v in zip(fixed_params, config)])
        ax1.set_title(f'K3 Infinity Norm vs {vary_param}\n{config_str}')
        ax1.set_xlabel(vary_param)
        ax1.set_ylabel('Infinity Norm')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()

        # Max Eigenvalue Plot with Error Bars
        ax2.errorbar(x, y_eig, yerr=y_eig_std, fmt='o', markersize=8, capsize=5)
        slope_eig, intercept_eig, r_value_eig, _, _ = linregress(np.log(x), np.log(y_eig))
        ax2.plot(x_line, np.exp(intercept_eig) * x_line**slope_eig, '--',
                label=f'slope={slope_eig:.2f}')
        
        ax2.set_title(f'K3 Max Eigenvalue vs {vary_param}\n{config_str}')
        ax2.set_xlabel(vary_param)
        ax2.set_ylabel('Max Eigenvalue')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

        with open(slopes_file, 'a') as f:
            f.write(f"{config_str} | {slope_inf:.3f} | {slope_eig:.3f} | {r_value_inf**2:.3f} | {len(x)}\n")

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
print("\nMultivariate Linear Regression Analysis in Log Space:") # perform multivariate regression
X = np.array([[np.log(d['L']), np.log(d['D_IN']), np.log(d['N']), np.log(d['M'])] for d in data])
y_inf = np.array([np.log(d['inf_norm']) for d in data])
y_eig = np.array([np.log(np.max(np.abs(d['mean_eigenvalues']))) for d in data])

reg_inf = LinearRegression().fit(X, y_inf)
reg_eig = LinearRegression().fit(X, y_eig)

print("\nInfinity Norm Regression coefficients:") # print regression results
print(f"L coefficient: {reg_inf.coef_[0]:.3f}")
print(f"D_IN coefficient: {reg_inf.coef_[1]:.3f}")
print(f"N coefficient: {reg_inf.coef_[2]:.3f}")
print(f"M coefficient: {reg_inf.coef_[3]:.3f}")
print(f"Intercept: {reg_inf.intercept_:.3f}")
print(f"R² score: {reg_inf.score(X, y_inf):.3f}")

print("\nMax Eigenvalue Regression coefficients:")
print(f"L coefficient: {reg_eig.coef_[0]:.3f}")
print(f"D_IN coefficient: {reg_eig.coef_[1]:.3f}")
print(f"N coefficient: {reg_eig.coef_[2]:.3f}")
print(f"M coefficient: {reg_eig.coef_[3]:.3f}")
print(f"Intercept: {reg_eig.intercept_:.3f}")
print(f"R² score: {reg_eig.score(X, y_eig):.3f}")

print("\nThis means the scaling relationships are approximately:") # show scaling relationships
print(f"inf_norm ∝ L^{reg_inf.coef_[0]:.3f} * D_IN^{reg_inf.coef_[1]:.3f} * N^{reg_inf.coef_[2]:.3f} * M^{reg_inf.coef_[3]:.3f}")
print(f"max_eigenvalue ∝ L^{reg_eig.coef_[0]:.3f} * D_IN^{reg_eig.coef_[1]:.3f} * N^{reg_eig.coef_[2]:.3f} * M^{reg_eig.coef_[3]:.3f}")
