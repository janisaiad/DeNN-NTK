# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from infinitewidth.kernel3_infinite import Kernel3Infinite # type: ignore

# %%
N = 10  # number of data points
D_IN = 20  # input dim
L_VALUES = np.linspace(2, 20, 18).astype(int)
RANDOM_SEED = 42
PATH_TO_PLOTS = "/home/janis/STG3A/deeperorwider/experiments/plots"
PATH_TO_DATA = "/home/janis/STG3A/deeperorwider/experiments/data"

# ensure paths exist
if not os.path.exists(PATH_TO_PLOTS):
    os.makedirs(PATH_TO_PLOTS)
if not os.path.exists(PATH_TO_DATA):
    os.makedirs(PATH_TO_DATA)

# %%
def generate_data(key, n_samples, n_features):
    """generate random data and normalize it"""
    rng = np.random.RandomState(key)
    data = rng.randn(n_samples, n_features)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

# %%
key_seed = RANDOM_SEED
mean_eigenvalues_per_L = []
std_eigenvalues_per_L = []
inf_norms_per_L = []

print("starting the analysis of the k3 tensor with the infinite width model...")
for L in L_VALUES:
    print(f"computing for l = {L} layers...")

    infinite_ntk = Kernel3Infinite(n_layers=L, n_outputs=1, a=1.0, b=1.0)
    data = generate_data(key_seed, N, D_IN)

    # compute the full k3 tensor
    k3_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):
                val = infinite_ntk._kernel_3_entry(data[i], data[j], data[k])
                # apply symmetry
                k3_tensor[i, j, k] = k3_tensor[i, k, j] = k3_tensor[j, i, k] = \
                k3_tensor[j, k, i] = k3_tensor[k, i, j] = k3_tensor[k, j, i] = val

    # compute infinity norm of the tensor
    inf_norm = np.max(np.abs(k3_tensor))
    inf_norms_per_L.append(inf_norm)

    # analyze eigenvalues of tensor slices
    all_slice_eigenvalues = []
    for k in range(N):
        slice_matrix = k3_tensor[:, :, k]
        eigenvalues = np.linalg.eigvalsh(slice_matrix)
        all_slice_eigenvalues.append(eigenvalues)
    
    # calculate mean and std for each eigenvalue order
    all_slice_eigenvalues = np.array(all_slice_eigenvalues) # shape (N, N)
    mean_eigenvalues_per_L.append(np.mean(all_slice_eigenvalues, axis=0))
    std_eigenvalues_per_L.append(np.std(all_slice_eigenvalues, axis=0))

# convert lists to numpy arrays for easier indexing
mean_eigenvalues_per_L = np.array(mean_eigenvalues_per_L)
std_eigenvalues_per_L = np.array(std_eigenvalues_per_L)
inf_norms_per_L = np.array(inf_norms_per_L)

# %% plot 1: infinity norm of k3 tensor vs. l
plt.figure(figsize=(10, 6))
plt.title(f"infinity norm of the k3 tensor vs. network depth (n={N}, d={D_IN})")
plt.plot(L_VALUES, inf_norms_per_L, 'o-')
plt.xlabel("number of layers (l)")
plt.ylabel("infinity norm")
plt.xticks(L_VALUES)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.savefig(PATH_TO_PLOTS + "/k3_inf_norm_vs_L_infinite.png")
print("k3 infinity norm plot saved to " + PATH_TO_PLOTS + "/k3_inf_norm_vs_L_infinite.png")

# %% plot 2: mean eigenvalues of k3 tensor slices vs. l
plt.figure(figsize=(12, 7))
plt.title(f"mean eigenvalues of k3 tensor slices vs. network depth (n={N}, d={D_IN})")

# plot each eigenvalue order with confidence bars
for k in range(N):
    means = mean_eigenvalues_per_L[:, k]
    stds = std_eigenvalues_per_L[:, k]
    p = plt.plot(L_VALUES, means, 'o-', label=f'λ_{k+1}')
    plt.fill_between(L_VALUES, means - stds, means + stds, alpha=0.2, color=p[0].get_color())

plt.xlabel("number of layers (l)")
plt.ylabel("mean eigenvalue of slice")
plt.yscale('log')
plt.xticks(L_VALUES)
plt.grid(True, which="both", ls="--")
plt.legend(title="eigenvalue order", bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # adjust layout to make room for legend
plt.savefig(PATH_TO_PLOTS + "/k3_slice_eigenvalues_vs_L_infinite.png")
print("k3 slice eigenvalues plot saved to " + PATH_TO_PLOTS + "/k3_slice_eigenvalues_vs_L_infinite.png")

# %% save data
output_data = {
    'l_values': L_VALUES,
    'inf_norms': inf_norms_per_L,
    'mean_eigenvalues': mean_eigenvalues_per_L,
    'std_eigenvalues': std_eigenvalues_per_L,
    'config': {'N': N, 'D_IN': D_IN, 'RANDOM_SEED': RANDOM_SEED}
}
np.save(PATH_TO_DATA + f"/k3_analysis_infinite_L_N={N}_D={D_IN}.npy", output_data)
print("k3 analysis data saved to " + PATH_TO_DATA + f"/k3_analysis_infinite_L_N={N}_D={D_IN}.npy")

