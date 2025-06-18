import numpy as np
import matplotlib.pyplot as plt
import os
from infinitewidth import InfiniteWidth

# %%
N = 100  # Nombre de points de données
D_IN = 10  # Dimension d'entrée
L_VALUES = np.linspace(2, 64, 10).astype(int)
RANDOM_SEED = 42
N_experiments = 10  # Pour le calcul de la moyenne des valeurs propres

# %%
def generate_data(key, n_samples, n_features):
    """Génère des données aléatoires et les normalise."""
    # Utilisation de np.random.RandomState pour la compatibilité
    rng = np.random.RandomState(key)
    data = rng.randn(n_samples, n_features)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

# %%
key_seed = RANDOM_SEED
eigenvalues_per_L = []

print("Démarrage de l'analyse des valeurs propres du NTK avec le modèle de largeur infinie...")
for L in L_VALUES:
    print(f"Calcul pour L = {L} couches...")
    eigenvalues_experiments = []
    
    # Pour la fonction d'activation ReLU, a=1, b=1 donne delta_phi=0.5
    # qui correspond à la non-linéarité de ReLU
    infinite_ntk = InfiniteWidth(n_layers=L, n_outputs=1, a=1.0, b=1.0)
    
    for exp in range(N_experiments):
        # Générer des données différentes pour chaque expérience
        data = generate_data(key_seed + exp, N, D_IN)
        
        # Calculer la matrice NTK théorique
        k_matrix = infinite_ntk.kernel_matrix(data)
        
        eigenvalues = np.linalg.eigvalsh(k_matrix)
        eigenvalues_experiments.append(eigenvalues)
    
    # Faire la moyenne sur les expériences
    mean_eigenvalues = np.mean(np.stack(eigenvalues_experiments), axis=0)
    eigenvalues_per_L.append(mean_eigenvalues)

if not os.path.exists('plots'):
    os.makedirs('plots')

# Tracer les histogrammes des valeurs propres
n_L = len(L_VALUES)
fig, axes = plt.subplots(1, n_L, figsize=(5 * n_L, 4), sharey=True)
fig.suptitle(f'Histogramme des valeurs propres moyennes du NTK théorique ({N_experiments} expériences)')
for i, L in enumerate(L_VALUES):
    ax = axes[i]
    ax.hist(eigenvalues_per_L[i], bins='auto', density=True)
    ax.set_title(f"Profondeur L = {L}")
    ax.set_xlabel("Valeur propre")
    ax.set_yscale('log')
axes[0].set_ylabel("Densité")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/ntk_infinite_eigenvalue_histograms.png")
print("Histogrammes des valeurs propres enregistrés dans plots/ntk_infinite_eigenvalue_histograms.png")

# Tracer la k-ième valeur propre en fonction de L
eigenvalues_per_L = np.array(eigenvalues_per_L)

plt.figure(figsize=(10, 6))
plt.title(f"Valeurs propres du NTK théorique en fonction de la profondeur du réseau ({N_experiments} expériences)")

# Tracer les plus grandes valeurs propres
for k in range(min(D_IN, N)):
    plt.plot(L_VALUES, eigenvalues_per_L[:, -(k+1)], 'o-', label=f'λ_{k+1}')
    
# Tracer la plus petite valeur propre
plt.plot(L_VALUES, eigenvalues_per_L[:, 0], 'o--', label=f'λ_min')

plt.xlabel("Nombre de couches (L)")
plt.ylabel("Valeur propre moyenne")
plt.xticks(L_VALUES)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("plots/kth_eigenvalue_vs_L_infinite.png")
print("Graphique de la k-ième valeur propre enregistré dans plots/kth_eigenvalue_vs_L_infinite.png")