import numpy as np
import matplotlib.pyplot as plt
import os
from infinitewidth import InfiniteWidth # type: ignore

# %%
N = 100  # number of data points
D_IN = 20  # input dim
L_VALUES = np.linspace(2, 200, 10).astype(int)
RANDOM_SEED = 42
PATH_TO_PLOTS = "/home/janis/STG3A/deeperorwider/experiments/plots"

# %%
def generate_data(key, n_samples, n_features):
    """generate random data and normalize it"""
    # use np.random.RandomState for compatibility
    rng = np.random.RandomState(key)
    data = rng.randn(n_samples, n_features)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

# %%
key_seed = RANDOM_SEED
eigenvalues_per_L = []

print("Starting the analysis of the eigenvalues of the NTK with the infinite width model...")
for L in L_VALUES:
    print(f"Computing for L = {L} layers...")
    eigenvalues_experiments = []

    # this remains relu
    infinite_ntk = InfiniteWidth(n_layers=L, n_outputs=1, a=1.0, b=1.0)
        
    data = generate_data(key_seed, N, D_IN)

    # we compute the NTK matrix with the infinite width model now
    k_matrix = infinite_ntk.kernel_matrix(data)
                
    eigenvalues = np.linalg.eigvalsh(k_matrix)
    eigenvalues_experiments.append(eigenvalues)
        
    
    mean_eigenvalues = np.mean(np.stack(eigenvalues_experiments), axis=0) # compute the mean of the eigenvalues
    eigenvalues_per_L.append(mean_eigenvalues)

if not os.path.exists(PATH_TO_PLOTS):
    os.makedirs(PATH_TO_PLOTS)

# we just! plot the histograms of the eigenvalues
n_L = len(L_VALUES)
fig, axes = plt.subplots(1, n_L, figsize=(5 * n_L, 4), sharey=True)
fig.suptitle(f'Histogram of the mean eigenvalues of the NTK with the infinite width model')
for i, L in enumerate(L_VALUES):
    ax = axes[i]
    eigenvalues = eigenvalues_per_L[i].copy()  # Make a copy to avoid modifying original
    eigenvalues = eigenvalues[eigenvalues != np.max(eigenvalues)]  # Remove max eigenvalue
    ax.hist(eigenvalues, bins=100, density=True)  # Plot histogram without max eigenvalue to avoid scale issues
    ax.set_title(f"Profondeur L = {L}")
    ax.set_xlabel("Valeur propre")
    ax.set_yscale('log')
    #ax.set_xscale('log')
axes[0].set_ylabel("Densité")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(PATH_TO_PLOTS + "/ntk_infinite_eigenvalue_histograms.png")
print("Histograms of the eigenvalues saved in " + PATH_TO_PLOTS + "/ntk_infinite_eigenvalue_histograms.png")

# we will plot the k-th eigenvalue as a function of L
eigenvalues_per_L = np.array(eigenvalues_per_L)

plt.figure(figsize=(10, 6))
plt.title(f"Eigenvalues of the NTK with the infinite width model as a function of the depth of the network")



# plot the largest eigenvalues without the largest one
for k in range(1, min(D_IN, N)): # we remove the largest that scale too much
    plt.plot(L_VALUES, eigenvalues_per_L[:, -(k+1)], 'o-', label=f'λ_{k+1}')

# and the smallest eigenvalue
plt.plot(L_VALUES, eigenvalues_per_L[:, 0], 'o--', label=f'λ_min')

plt.xlabel("Nombre de couches (L)")
plt.ylabel("Valeur propre moyenne")
plt.xticks(L_VALUES)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(PATH_TO_PLOTS + "/kth_eigenvalue_vs_L_infinite.png")
print("Graphique de la k-ième valeur propre enregistré dans " + PATH_TO_PLOTS + "/kth_eigenvalue_vs_L_infinite.png")


# %%
plt.figure(figsize=(10, 6))
plt.title(f"Eigenvalues of the NTK with the infinite width model as a function of the depth of the network")
# plot the largest eigenvalue
plt.plot(L_VALUES, eigenvalues_per_L[:, -1], 'o-', label=f'λ_max')
plt.xlabel("Nombre de couches (L)")
plt.ylabel("Valeur propre moyenne")
plt.xticks(L_VALUES)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(PATH_TO_PLOTS + "/largest_eigenvalue_vs_L_infinite.png")
print("Graphique de la k-ième valeur propre enregistré dans " + PATH_TO_PLOTS + "/largest_eigenvalue_vs_L_infinite.png")


