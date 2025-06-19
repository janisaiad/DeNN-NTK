# %%
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
import os
import neural_tangents as nt
from neural_tangents import stax
from infinitewidth.infinitewidth import InfiniteWidth

N_VALUES = [8, 10,16, 25, 32,40,50 ,64,80,100,110, 128,150,180,200,230,256]  # we use different numbers of data points
D_IN_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000]  # we test different input dimensions  
M_VALUES = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]  # we vary network widths
L_VALUES = np.arange(2, 60, 2)  # we test network depths
N_EXPERIMENTS = 10  # number of experiments per configuration
RANDOM_SEED = 42

PATH_TO_DATA = "/home/janis/STG3A/deeperorwider/experiments/data/large_ntk_corrections"
os.makedirs(PATH_TO_DATA, exist_ok=True)

def generate_data(key, n_samples, n_features):
    data = random.normal(key, (n_samples, n_features))
    norm = jnp.linalg.norm(data, axis=1, keepdims=True)
    return data / norm  # normalize data points to unit norm

experiments = []
for i1, N in enumerate(N_VALUES):
    for i2, D_IN in enumerate(D_IN_VALUES):
        for i3, M in enumerate(M_VALUES):
            for i4, L in enumerate(L_VALUES):
                complexity = i1 + i2 + i3 + 0.2*i4  # score to order experiments by computational complexity
                experiments.append((complexity, N, D_IN, M, L))

experiments.sort(key=lambda x: x[0])

key_seed = RANDOM_SEED
print("Starting analysis of NTK correction term...")

for complexity, N, D_IN, M, L in experiments:
    filename = f"ntk_correction_N{N}_D{D_IN}_M{M}_L{L}.npy"
    
    if not os.path.isfile(os.path.join(PATH_TO_DATA, filename)):
        print(f"\nComputing for N={N}, D_IN={D_IN}, M={M}, L={L}...")
        
        spectral_radii = []
        for exp in range(N_EXPERIMENTS):
            key = jax.random.PRNGKey(key_seed)
            key_seed += 1
            
            # we split the key for data generation and model initialization
            data_key, model_key = random.split(key)
            
            data = generate_data(data_key, N, D_IN)
            
            layers = []
            layers.append(stax.Dense(M, W_std=jnp.sqrt(2), b_std=0.0, parameterization='ntk'))
            layers.append(stax.Relu())
            
            for _ in range(L - 1):
                layers.append(stax.Dense(M, W_std=jnp.sqrt(2), b_std=0.0, parameterization='ntk'))
                layers.append(stax.Relu())
                
            layers.append(stax.Dense(1, W_std=jnp.sqrt(2), b_std=0.0, parameterization='ntk'))
            
            init_fn, apply_fn, _ = stax.serial(*layers)
            params = init_fn(model_key, data.shape)[1]
            ntk_fn = nt.empirical_ntk_fn(apply_fn)
            K = ntk_fn(data, None, params)  # compute empirical NTK
            
            inf_model = InfiniteWidth(n_layers=L, n_outputs=1)
            K_prime = inf_model.kernel_matrix(data)  # compute infinite-width NTK
            
            correction = M * (K - K_prime)  # compute finite-width correction term
            eigenvalues = np.linalg.eigvalsh(correction)
            spectral_radius = np.max(np.abs(eigenvalues))  # get spectral radius of correction
            spectral_radii.append(spectral_radius)
            
        mean_radius = np.mean(spectral_radii)
        std_radius = np.std(spectral_radii)
        
        output_data = {
            'N': N,
            'D_IN': D_IN, 
            'M': M,
            'L': L,
            'mean_spectral_radius': mean_radius,
            'std_spectral_radius': std_radius,
            'RANDOM_SEED': RANDOM_SEED
        }
        
        np.save(os.path.join(PATH_TO_DATA, filename), output_data)
        print(f"Data saved to {PATH_TO_DATA}/{filename}")
        print(f"Mean spectral radius: {mean_radius:.4f} ± {std_radius:.4f}")
        
    else:
        print(f"Skipping {filename} because it already exists")

# %%
