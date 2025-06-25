from infinitewidth import NtkInfiniteWidth
import jax
import jax.numpy as jnp
from jax import random

import os


# we will see the eigenvectors distribution of the NTK matrix
# I guess there is like a uniform distribution of the eigenvectors in the S^(N-2), which is the sphere section orthogonal to the max eigenvalue

N_VALUES = [8, 10,16, 25, 32,40,50 ,64,80,100,110, 128,150,180,200,230,256]  # we use different numbers of data points
D_IN_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000]  # we test different input dimensions  
M_VALUES = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]  # we vary network widths
L_VALUES = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000]  # we test network depths in log space
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
                complexity = i1 + i2 + i3 + i4  # score to order experiments by computational complexity
                experiments.append((complexity, N, D_IN, M, L))

experiments.sort(key=lambda x: x[0])

key_seed = RANDOM_SEED
print("Starting analysis of NTK correction term...")

for complexity, N, D_IN, M, L in experiments:
    filename = f"ntk_correction_N{N}_D{D_IN}_M{M}_L{L}.npy"
    
    if not os.path.isfile(os.path.join(PATH_TO_DATA, filename)):
        try:
            print(f"\nComputing for N={N}, D_IN={D_IN}, M={M}, L={L}...")
            
            all_eigenvalues = []
            all_eigenvectors = []
            for exp in range(N_EXPERIMENTS):
                key = jax.random.PRNGKey(key_seed)
                key_seed += 1
                
                data_key, model_key = random.split(key)  # we split the key for data generation and model initialization
                
                data = generate_data(data_key, N, D_IN)
                
                inf_model = NtkInfiniteWidth(n_layers=L, n_outputs=1)
                K_prime = inf_model.kernel_matrix(data)  # we compute infinite-width NTK
                eigenvalues, eigenvectors = jnp.linalg.eigh(K_prime)  # we compute eigenvalues and eigenvectors
                
                all_eigenvalues.append(eigenvalues)
                all_eigenvectors.append(eigenvectors)
            
            output_data = {
                'N': N,
                'D_IN': D_IN, 
                'M': M,
                'L': L,
                'eigenvalues': np.array(all_eigenvalues),  # shape (n_experiments, N)
                'eigenvectors': np.array(all_eigenvectors),  # shape (n_experiments, N, N)
                'RANDOM_SEED': RANDOM_SEED
            }
            
            np.save(os.path.join(PATH_TO_DATA, filename), output_data)
            print(f"Data saved to {PATH_TO_DATA}/{filename}")
            print(f"Completed experiment for N={N}, D_IN={D_IN}, M={M}, L={L}")
        except Exception as e:
            print(f"Error for N={N}, D_IN={D_IN}, M={M}, L={L}: {e}")
            continue
    else:
        print(f"Skipping {filename} because it already exists")

# %%
