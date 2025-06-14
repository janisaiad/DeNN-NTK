## Presentation Script: Spectral Analysis of the Neural Tangent Kernel

Today I'll present what I've learned about the spectral analysis of Neural Tangent Kernels and Sobolev training. This work synthesizes several key papers and my own analysis to understand how spectral properties control learning dynamics, with a particular focus on establishing optimization bounds under the NTK perspective.

### Introduction and Core Objectives

So first, let me set up the three fundamental objectives we're trying to address:

1. **Eigenvalue scaling laws**: We want to derive decay rates μ_ℓ ~ ℓ^(-α) for NTK operator eigenvalues
2. **Spectral impact on learning**: Understanding how these spectral properties actually determine learning dynamics
3. **Matrix vs. Operator relationship**: This is crucial - we need to analyze scaling laws for discrete matrix eigenvalues with respect to network depth l and data size n

The NTK is defined as this inner product of gradients:
K^∞(x_i, x_j) = ⟨∂f(x_i; θ)/∂θ, ∂f(x_j; θ)/∂θ⟩

### Motivation: Towards Optimization Bounds under NTK Regime

Now, why is this analysis important? The primary objective is to establish optimization bounds under the NTK perspective for neural network regression and solving partial differential equations. What I mean by optimization bounds is something like showing that with probability 1-δ, after time t of training, ||f_t - f*||_2 ≤ ε for appropriate choices of network width and depth.

Recent work, notably by Yang & He, has already established robust generalization and approximation bounds for deep super-ReLU networks. Their comprehensive analysis demonstrates that deep networks can achieve optimal generalization rates under Sobolev loss while simultaneously enabling efficient approximation of functions in Sobolev spaces through architectural depth.

However, an important gap remains: optimization bounds are largely unexplored in the feature learning regime and after the initialization. While we understand how well these networks can generalize and what functions they can represent, the fundamental question of how efficiently they can be trained remains unanswered from a rigorous theoretical perspective.

This NTK perspective proves particularly relevant for PDE applications where fine control of Fourier components determines numerical solution quality, but optimization bounds remain largely unexplored under Sobolev loss settings. This is directly relevant for Deep Ritz Method, Deep Galerkin Method, and Physics-Informed Neural Networks (PINNs) where spectral properties determine convergence rates and solution accuracy.

### Research Goals and Factorization Strategy

Our spectral analysis aims to fill this theoretical gap by establishing comprehensive understanding of how NTK spectral properties determine optimization convergence rates in practice. We seek to characterize the intricate relationships between network depth, data size, and optimization difficulty through matrix conditioning analysis.

Crucially, I've proved that Sobolev training modifies the underlying spectrum in a disentangled manner - the composite operator T_s = K^∞ ∘ P_s factorizes independently on both sphere and torus domains, enabling precise control of frequency components.

The key insight is that the NTK data matrix spectrum remains separable as a product λ_i(KP_s) = λ_i(K) · λ_i(P_s), allowing us to disentangle architectural effects (through K) from loss function effects (through P_s). This spectral factorization dictates the training trajectory, with architectural choices affecting λ_i(K) - including depth L, width N, and sophisticated profiles like m_ℓ = m·ℓ² for MLPs at edge of chaos - while Sobolev order s independently controls λ_i(P_s).

### Critical Distinction: Matrix vs Operator

This is super important and often overlooked. We have two different objects:
- **NTK Matrix**: The discrete sampled version K ∈ ℝ^(n×n) with entries K_ij = K^∞(x_i, x_j)
- **NTK Operator**: The continuous integral operator (Lf)(x) = ∫K^∞(x,y)f(y)dy

The eigenvalues of the sampled NTK matrix are NOT the same as the NTK operator eigenvalues. This is a fundamental point that many people miss.

Now, why do we focus on spherical domains? It's mainly for computational tractability - we get explicit spectral decompositions through spherical harmonic symmetrization. But there's a cost: there's no uniform sampling measure on the sphere, so we have to analyze the spectrum indirectly through inverse cosine distance matrix approximations. This limitation actually drives us to explore alternative domains later.

### NTK Matrix Structure and Deep Networks

Let me show you what this looks like concretely. For 3 data points, you get this matrix structure where each entry depends on the kernel evaluation between points. For general depth l networks at Edge of Chaos initialization, we have this complex formula involving the cosine map ρ.

For the simple 2-layer ReLU case, it reduces to:
k(x_i,x_j) = x_i^T x_j · arccos(-⟨x_i,x_j⟩) + √(1-⟨x_i,x_j⟩²)

The key spectral results are:
- Condition number: κ(K^∞) ~ 1 + n/3 + O(nξ/l)
- Eigenvalue distribution: λ_min ~ 3l/(4n), λ_max ~ 3l/4 ± ξ where ξ ~ log(l)

So both eigenvalues scale linearly with depth, but the condition number grows with n. That's why deeper networks improve conditioning but with diminishing returns.

For deep networks, we have this eigenvalue decay μ_k ~ C(d,L)k^(-d) where C(d,L) grows quadratically with L. But there's a really nice result about inverse cosine distance matrices - the NTK matrix has this near-affine behavior: K^∞ ≈ A·W_l + B.

This relationship enables indirect analysis of NTK spectral properties through simpler geometric matrices, which is computationally much more tractable.

### Deep Narrow Networks

This is a fascinating direction. For deep narrow networks, the scaled NTK converges to a Gaussian process limit. The comparison with two-layer kernels is striking:
- Two-layer: μ_ℓ^(2) ~ ℓ^(-(d+1))
- Deep narrow: μ_ℓ^(dn) ~ C(L,d)ℓ^(-d) where C(L,d) ∝ L

So depth partially compensates for the poor conditioning of two-layer kernels. The spectrum is flatter: κ(K^(dn)) ≈ κ(K^(2))/L.

There are several research perspectives here - architectural modifications like unit concatenation and skip connections, initialization studies including the β → 0 limit, and theoretical extensions. However, some of these directions like the extension of Hayou & Yang's work on ResNets use a mean field approach that's not easily generalizable.

### Alternative Domains: My Proper Contributions

Now, this is where I think there's a lot of potential for future work, and these questions were told to be interesting but not to be investigated now in the literature. To apply integration by parts well, we need to have a domain without a boundary!

**Gaussian domain**: Uses Ornstein-Uhlenbeck operator with Hermite polynomial eigenfunctions. The Gaussian measure introduces natural regularization through exponential decay. However, the NTK is not translation-invariant K^∞(x, y) = k(x - y) but it is invariant under rotations. The composite spectrum becomes λ_α ~ k̂(|α|^(1/2)) · (1 + |α|)^s.

**Toroidal domain**: This is really promising and represents a proper contribution! Suppose you have a dataset over [0,1]^d and you want to learn a function on this domain. You can use a neural network with periodic boundary conditions - you glue opposite faces of the unit cube [0,1]^d to get a torus.

The key advantage is exact orthogonality of Fourier modes under uniform sampling:
(1/N) Σ e^(2πik₁j/N) e^(-2πik₂j/N) = δ_{k₁,k₂ mod N}

This means we can explicitly construct all eigenvectors and use FFT for O(N^d log N) eigenvalue computation! The computational cost with FFT enables O(N^d log N) for matrix-vector products. But I've made computations by hand - we can do very fast experiments for a whole range of initialization/architecture/functions and this case is not yet validated in the literature.

### Sobolev Training Framework

Now here's where it gets really interesting. The key innovation in Sobolev training is modifying the standard L² loss to incorporate high-order derivatives. 

The Sobolev operator P_s is defined in Fourier space as:
P_s = Σ(1+ℓ)^(2s) P_{ℓ,p}

The beautiful thing is that both the NTK operator K^∞ and Sobolev operator P_s share spherical harmonics as eigenfunctions due to rotational invariance. This means they commute: [K^∞, P_s] = 0.

### The Five Main Proofs

I've worked through five key theoretical results:

**Proof 1**: Shows that Sobolev loss can be written as a fractional Laplacian. The key insight is using spherical harmonic expansion where the Laplacian acts as (-Δ)^(1/2) Y_{ℓ,p} = √(ℓ(ℓ + d - 2)) Y_{ℓ,p}.

**Proof 2**: Under Sobolev training, the learning operator becomes T_s = K^∞ ∘ (I + (-Δ)^(1/2))^s. This comes from the chain rule when you replace the L² loss with the Sobolev loss.

**Proof 3**: The eigenvalues of the composite operator are just products: μ_ℓ^(T_s) = μ_ℓ^(K) · (1 + √(ℓ(ℓ + d - 2)))^s.

**Proof 4**: The discrete matrices K and P_s commute: KP_s = P_sK. This is because they're both expressed in terms of the same spherical harmonic projectors.

**Proof 5**: The asymptotic scaling laws give λ_ℓ ~ ℓ^(s-d). This is the critical result - the spectral behavior depends on whether s < d (regularizing), s = d (critical), or s > d (amplifying high frequencies).

### Perturbation Theory Framework for Finite-Width Effects

Now, let me discuss the advanced mathematical machinery needed for optimization bounds. We employ a systematic expansion in powers of 1/n for computation of training dynamics, with experimental validation of O(1/n) training dynamics under gradient flow.

Finite-width networks require analysis beyond leading-order asymptotics. Diagrammatic techniques reveal correlation function scaling at infinite width. There are three key modifications to infinite-width behavior:
- Initial NTK Θ₀ receives width-dependent corrections
- Network updates become nonlinear in learning rate  
- NTK becomes time-dependent during training

The perturbative expansion takes the form: H = H₀ + (1/n)H₁ + (1/n²)H₂ + ..., where each H_k term captures neuron interactions at finite width, and the expansion controls finite-size effects in optimization.

### Finite-Width Corrections and NTK Hierarchy

The finite-width NTK expansion takes this explicit form:

Θ^NTH(x₁, x₂) = Θ(x₁, x₂) + n^(-1)𝔼[𝒪^(1)_{2,0}(x₁, x₂)] - n^(-1)𝔼[𝒪^(1)_{3,0}(x₁, x₂, x⃗)Θ^(-1)(x⃗, x⃗)f^(0)_0(x⃗)] + n^(-1)y⃗^T Θ^(-1)(x⃗, x⃗)𝔼[𝒪^(1)_{4,0}(x₁, x₂, x⃗, x⃗)] Θ^(-1)(x⃗, x⃗)y⃗ + higher order terms

These corrections capture how the idealized infinite-width behavior degrades in practice. The tensors 𝒪^(1)_{4,0} and 𝒪^(1)_{3,0} represent higher-order interaction terms that require detailed investigation to understand their impact on spectral properties and optimization dynamics.

What we already know is that we have spectrum estimates for the NTK matrix in infinite-width, so we need to treat the other terms. Higher orders capture complex neuronal interactions at finite width. Then we fix parameter budget P, express corrections in terms of depth L and width N = P/L, and analytically determine whether increasing depth or width provides better optimization guarantees.

### Unified Framework: My Proper Contribution

The reconciliation between geometric and functional views comes through shared algebraic structure. Both K and P_s are rotationally invariant, so they commute and are simultaneously diagonalizable. The eigenvalues of the composite operator are simply products:
λ_i(KP_s) = λ_i(K) · λ_i(P_s)

I've also shown that P_s has a zonal kernel representation with explicit Gegenbauer polynomial expansions, which enables fast multipole methods. The Sobolev matrix P_s can be written as a zonal kernel (P_s)_ij = p_s(⟨x_i, x_j⟩) with eigenvalue bounds 1 ≤ λ_i(P_s) ≤ (1 + ℓ_max)^s and condition number κ(P_s) = (1 + ℓ_max)^s.

Our proposed strategy is:
1. Approximate NTK spectrum via inverse cosine distance matrix
2. Analyze Sobolev operator spectrum  
3. Combine via spectrum product and validate experimentally

### Component-wise Investigation and Beyond Lazy Training

From that factorization T_s = K^∞ ∘ P_s, we can simplify research by enabling independent investigation of each component. For a specific domain, we can investigate the spectrum of P_s via experimental frameworks with Fourier analysis.

Importantly, from that investigation of NTK matrix K^∞ we can extend to the feature learning regime. Previous work like Banerjee et al. bounds λ_min(K) but doesn't assume NTK regime operation. Our approach explicitly assumes NTK regime validity and investigates finite-width corrections within this setting, opening theoretical questions about interplay between kernel constancy, spectral properties, and architectural parameters.

### Computational Implementation Strategy

For the toroidal domain, the implementation strategy leverages Kronecker structure. The d-dimensional DFT decomposes as F_d = F₁ ⊗ F₁ ⊗ ... ⊗ F₁. The algorithm becomes:
1. Forward FFT: f̂ = F_d f using d successive 1D FFTs
2. Spectral multiplication: ĝ_k = (1 + ||k||²)^s f̂_k  
3. Inverse FFT: g = F_d* ĝ using d successive 1D inverse FFTs

This goes from O(N^(2d)) naive approach to O(d·N^d log N) with FFT, which is a massive improvement.

### Research Roadmap and Immediate Challenges

**Near-term objectives:**
- Compute the NTK perturbation matrix to get immediate spectrum bounds for finite width
- Unify initialization schemes (for example no bias and EOC) and architectural assumptions  
- Study narrow NTK behavior to identify complexifications

**Long-term goals:**
- P_s matrix is a zonal kernel, can be investigated trying to get a better conditioning number multiplicative constant
- Develop more general theory for deep narrow networks (for other initialization without UAT)

There are some immediate simple challenges that are not to be investigated now:
- Unified framework: Different papers with varying domains, initializations, architectures, activations
- Extension of harmonic analysis to general spaces L²(γ)
- Extension from ReLU to general inhomogeneous activations
- Experimental validation: Systematic verification of theoretical predictions because some of the results are not yet validated

The key insight from Sobolev perspective is that we can tune the exponent s as a function of data size n. When n grows, κ(K) ~ n deteriorates, but multiplying by P_s amplifies high-frequency modes and can flatten the composite spectrum. Making s larger for large n counterbalances K's conditioning. However, we still lack a precise estimate of κ(P_s) - an important problem that warrants future investigation.

### Key Takeaways and Impact

The main impact of this work is that we've established a systematic framework for understanding optimization bounds in the NTK regime. The factorization T_s = K^∞ ∘ P_s allows us to separate architectural effects from regularization effects, making the analysis much more tractable.

Our perturbation theory approach provides a principled way to understand how finite-width effects degrade optimization performance. This is crucial because real networks always have finite width, and the infinite-width analysis is just our starting point.

The deeper-versus-wider question can now be addressed analytically by fixing a parameter budget P and comparing how depth L versus width N = P/L affects the smallest eigenvalue λ_min, which controls convergence rates.

For PDE applications, this framework is particularly valuable because we can now predict and control which frequency components will be learned effectively by tuning the Sobolev parameter s. This level of control over the learning dynamics is exactly what we need for high-quality numerical solutions.

The experimental validation on the toroidal domain provides a concrete path forward for testing these theoretical predictions at scale, with computational advantages that make large-scale studies feasible.

This represents a significant step toward completing the theoretical picture of neural networks - we now have generalization bounds, approximation bounds, and a clear path toward optimization bounds, all within a unified spectral framework. The convergence of spectral analysis, harmonic analysis, and neural network theory really opens up exciting possibilities for understanding and controlling learning dynamics through spectral properties.
