from infinitewidth import NtkInfiniteWidth
try:
    import os
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # we force GPU usage globally

    import jax
    jax.config.update('jax_platform_name', 'gpu')  # we configure JAX for GPU
except:
    import os
    import jax
    print("No GPU found, using CPU")
    pass
    

import jax.numpy as jnp
from jax import random
import numpy as np  # we add numpy import
import time  # we add time import for measuring computation time

# we will see the eigenvectors distribution of the NTK matrix
# I guess there is like a uniform distribution of the eigenvectors in the S^(N-2), which is the sphere section orthogonal to the max eigenvalue

N_VALUES = [8, 10,16, 25, 32,40,50 ,64,80,100,110, 128,150,180,200,230,256,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]  # we use different numbers of data points
D_IN_VALUES = [20, 50, 100, 200, 500, 1000]  # we test different input dimensions  
M_VALUES = [10,20,50,100,200,500,1000,2000,5000]  # we vary network widths
L_VALUES = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000]  # we test network depths in log space
N_EXPERIMENTS = 100  # number of experiments per configuration
RANDOM_SEED = 42


import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
PATH_TO_DATA = os.path.join(PROJECT_ROOT, "experiments", "data", "eigen")

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
                complexity = i1 + i2 + i3 + i4  # score to order experiments by computational complexity
                experiments.append((complexity, N, D_IN, M, L))

experiments.sort(key=lambda x: x[0])

key_seed = RANDOM_SEED
print("Starting analysis of NTK correction term...")

start_time_total = time.time()  # we track total computation time
force_recompute = True

for complexity, N, D_IN, M, L in experiments:
    filename_eigenvalues = f"values/ntk_eigenvalues_N{N}_D{D_IN}_M{M}_L{L}.npy"
    filename_eigenvectors = f"vectors/ntk_eigenvectors_N{N}_D{D_IN}_M{M}_L{L}.npy"
    
    if force_recompute or (not (os.path.isfile(os.path.join(PATH_TO_DATA, filename_eigenvalues))) or not (os.path.isfile(os.path.join(PATH_TO_DATA, filename_eigenvectors)))):
        try:
            print(f"\nComputing for N={N}, D_IN={D_IN}, M={M}, L={L}...")
            start_time_config = time.time()  # we track time for each configuration
            small_exp_time = time.time()
            all_eigenvalues = []
            all_eigenvectors = []
            for exp in range(N_EXPERIMENTS):
                  # we track time for each experiment
                
                key = jax.random.PRNGKey(key_seed)
                key_seed += 1
                
                data_key, model_key = random.split(key)  # we split the key for data generation and model initialization
                
                data = generate_data(data_key, N, D_IN)
                
                inf_model = NtkInfiniteWidth(n_layers=L, n_outputs=1)
                K_prime = inf_model.kernel_matrix(data)  # we compute infinite-width NTK
                eigenvalues, eigenvectors = jnp.linalg.eigh(K_prime)  # we compute eigenvalues and eigenvectors
                
                # the largest eigenvalue is the last one
                all_eigenvalues.append(eigenvalues)
                all_eigenvectors.append(eigenvectors)
                if exp % 10 == 9:
                    time_now = time.time()
                    print(f"Experiment {exp+1}/{N_EXPERIMENTS} completed in {time_now - small_exp_time:.2f} seconds")
                    small_exp_time = time.time()
            end_time_exp = time.time()
            exp_time = end_time_exp - start_time_config
            print(f"Experiment {exp+1}/{N_EXPERIMENTS} completed in {exp_time:.2f} seconds")
            
            output_data_eigenvalues = {
                'N_EXPERIMENTS': N_EXPERIMENTS,
                'N': N,
                'D_IN': D_IN, 
                'M': M,
                'L': L,
                'eigenvalues': jnp.array(all_eigenvalues),  # shape (n_experiments, N)
                'RANDOM_SEED': RANDOM_SEED
            }
            
            output_data_eigenvectors = {
                'N_EXPERIMENTS': N_EXPERIMENTS,
                'N': N,
                'D_IN': D_IN, 
                'M': M,
                'L': L,
                'eigenvectors': jnp.array(all_eigenvectors),  # shape (n_experiments, N, N)
                'RANDOM_SEED': RANDOM_SEED
            }
            # in this data, the largest eigenvector is the last one, we can access them with all_eigenvectors[:, -1, :]
            np.save(os.path.join(PATH_TO_DATA, filename_eigenvalues), output_data_eigenvalues)  # we use numpy save
            np.save(os.path.join(PATH_TO_DATA, filename_eigenvectors), output_data_eigenvectors)  # we use numpy save
            
            config_time = time.time() - start_time_config  # we calculate configuration time
            print(f"Data saved to {PATH_TO_DATA}/{filename_eigenvalues} and {PATH_TO_DATA}/{filename_eigenvectors}")
            print(f"Completed experiment for N={N}, D_IN={D_IN}, M={M}, L={L} in {config_time:.2f} seconds")
        except Exception as e:
            print(e)
            print(f"Error for N={N}, D_IN={D_IN}, M={M}, L={L}: {e}")
            continue
    else:
        print(f"Skipping {filename_eigenvalues} and {filename_eigenvectors} because they already exist")

total_time = time.time() - start_time_total  # we calculate total computation time
print(f"\nTotal computation time: {total_time:.2f} seconds")

# %%
