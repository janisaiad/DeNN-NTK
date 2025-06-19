import jax
import jax.numpy as jnp
import numpy as np
import os
import time
from finitewidth.kernel3_empirical import Kernel3Empirical

# --- Configuration ---
N_VALUES = [8, 16, 32, 64, 128, 256]  # we use different numbers of data points
D_IN_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000]  # we test different input dimensions  
M_VALUES = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]  # we vary network widths
L_VALUES = np.arange(2, 20, 2)  # we test network depths
RANDOM_SEED = 42

PATH_TO_DATA = "/home/janis/STG3A/deeperorwider/experiments/data/large"
os.makedirs(PATH_TO_DATA, exist_ok=True)

# --- Activation Function ---
def relu(x):
    return jnp.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)

# --- Data and Network Initialization ---
def generate_data(key, n_samples, n_features):
    """Generate random data and normalize it."""
    rng = jax.random.PRNGKey(key)
    data = jax.random.normal(rng, (n_samples, n_features))
    norm = jnp.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

def init_network(key, L, d_in, m):
    """Initialize network weights."""
    keys = jax.random.split(key, L + 1)
    weights = []
    weights.append(jax.random.normal(keys[0], (m, d_in)))
    
    for i in range(1, L):
        weights.append(jax.random.normal(keys[i], (m, m)))
    weights.append(jax.random.normal(keys[L], (m,)))
    return weights

def compute_features_and_derivatives(weights, data):
    """Perform forward pass to get feature maps and derivatives."""
    H = len(weights) - 1  # we get number of hidden layers
    m = weights[0].shape[0]  # we get width

    feature_maps = {0: [x for x in data]}  # we initialize with input data
    sigma_derivatives = {}
    current_activations = feature_maps[0]

    for l in range(1, H + 1):
        W = weights[l-1]
        pre_activations = [W @ x for x in current_activations]  # we compute pre-activations
        sigma_derivatives[l] = [jnp.diag(relu_prime(pre_act)) for pre_act in pre_activations]  # we store derivatives
        current_activations = [(1/jnp.sqrt(m)) * relu(pre_act) for pre_act in pre_activations]  # we apply activation
        feature_maps[l] = current_activations
        
    return feature_maps, sigma_derivatives

# --- Main Experiment Loop ---
key_seed = RANDOM_SEED

print("Starting large-scale analysis of the empirical K3 tensor...")

experiments = []
for i1, N in enumerate(N_VALUES):
    for i2, D_IN in enumerate(D_IN_VALUES):
        for i3, M in enumerate(M_VALUES):
            for i4, L in enumerate(L_VALUES):
                complexity = 2*i1 + 0.5*i2 + 0.5*i3 + i4  # we compute complexity score, penalize a lot N
                experiments.append((complexity, N, D_IN, M, L))

experiments.sort(key=lambda x: x[0])  # we sort by complexity score (1st coordinate)

for complexity, N, D_IN, M, L in experiments:
    # if the npy file exists, we skip the computation
    filename = f"k3_analysis_N{N}_D{D_IN}_M{M}_L{L}.npy"
    if not(os.path.isfile(os.path.join(PATH_TO_DATA, filename))): # we check if the file exists
            
        start_time = time.time()  # we start timing
        
        print(f"\nComputing for N={N}, D_IN={D_IN}, M={M}, L={L}...")
        
        key = jax.random.PRNGKey(key_seed)
        key_seed += 1
        weights = init_network(key, L, D_IN, M)  # we initialize network
        data = generate_data(key_seed, N, D_IN)  # we generate input data
        
        feature_maps, sigma_derivatives = compute_features_and_derivatives(weights, data)  # we do forward pass

        k3_computer = Kernel3Empirical(weights, sigma_derivatives, feature_maps)  # we setup k3 computer
        k3_tensor = np.zeros((N, N, N))  # we initialize tensor
        
        for i in range(N):
            for j in range(i, N):
                for k in range(j, N):
                    val = k3_computer.kernel3(i, j, k)  # we compute tensor entry
                    k3_tensor[i, j, k] = k3_tensor[i, k, j] = k3_tensor[j, i, k] = \
                    k3_tensor[j, k, i] = k3_tensor[k, i, j] = k3_tensor[k, j, i] = val  # we fill symmetric entries

        inf_norm = M * np.max(np.abs(k3_tensor))  # we compute infinity norm
        
        all_slice_eigenvalues = []
        for k in range(N):
            slice_matrix = k3_tensor[:, :, k]  # we get matrix slice
            eigenvalues = np.linalg.eigvalsh(slice_matrix)  # we compute eigenvalues
            all_slice_eigenvalues.append(eigenvalues)
        
        all_slice_eigenvalues = M * np.array(all_slice_eigenvalues)  # we scale eigenvalues
        mean_eigenvalues = np.mean(all_slice_eigenvalues, axis=0)  # we compute mean
        std_eigenvalues = np.std(all_slice_eigenvalues, axis=0)  # we compute std

        output_data = {
            'N': N,
            'D_IN': D_IN,
            'M': M,
            'L': L,
            'inf_norm': inf_norm,
            'mean_eigenvalues': mean_eigenvalues,
            'std_eigenvalues': std_eigenvalues,
            'RANDOM_SEED': RANDOM_SEED
        }
        
        filename = f"k3_analysis_N{N}_D{D_IN}_M{M}_L{L}.npy"
        np.save(os.path.join(PATH_TO_DATA, filename), output_data)  # we save results
        
        end_time = time.time()  # we end timing
        compute_time = end_time - start_time
        
        print(f"Computation time: {compute_time:.2f} seconds")
        print(f"Data saved to {PATH_TO_DATA}/{filename}")

        
    else:
        print(f"Skipping {filename} because it already exists")
        
    