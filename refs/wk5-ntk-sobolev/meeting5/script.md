## Presentation Script: Spectral Analysis of the Neural Tangent Kernel

Today I'll present what I've learned about the spectral analysis of Neural Tangent Kernels and Sobolev training. This work synthesizes several key papers and my own analysis to understand how spectral properties control learning dynamics.

### Introduction and Core Objectives

So first, let me set up the three fundamental objectives we're trying to address:

1. **Eigenvalue scaling laws**: We want to derive decay rates μ_ℓ ~ ℓ^(-α) for NTK operator eigenvalues
2. **Spectral impact on learning**: Understanding how these spectral properties actually determine learning dynamics
3. **Matrix vs. Operator relationship**: This is crucial - we need to analyze scaling laws for discrete matrix eigenvalues with respect to network depth l and data size n

The NTK is defined as this inner product of gradients:
K^∞(x_i, x_j) = ⟨∂f(x_i; θ)/∂θ, ∂f(x_j; θ)/∂θ⟩

Now, why do we focus on spherical domains? It's mainly for computational tractability - we get explicit spectral decompositions through spherical harmonic symmetrization. But there's a cost: there's no uniform sampling measure on the sphere, so we have to analyze the spectrum indirectly through inverse cosine distance matrix approximations. This limitation actually drives us to explore alternative domains later.

### Critical Distinction: Matrix vs Operator

This is super important and often overlooked. We have two different objects:
- **NTK Matrix**: The discrete sampled version K ∈ ℝ^(n×n) with entries K_ij = K^∞(x_i, x_j)
- **NTK Operator**: The continuous integral operator (Lf)(x) = ∫K^∞(x,y)f(y)dy

The eigenvalues of the sampled NTK matrix are NOT the same as the NTK operator eigenvalues. This is a fundamental point that many people miss.

### NTK Matrix Structure

Let me show you what this looks like concretely. For 3 data points, you get this matrix structure where each entry depends on the kernel evaluation between points. For general depth l networks at Edge of Chaos initialization, we have this complex formula involving the cosine map ρ.

For the simple 2-layer ReLU case, it reduces to:
k(x_i,x_j) = x_i^T x_j · arccos(-⟨x_i,x_j⟩) + √(1-⟨x_i,x_j⟩²)

The key spectral results are:
- Condition number: κ(K^∞) ~ 1 + n/3 + O(nξ/l)
- Eigenvalue distribution: λ_min ~ 3l/(4n), λ_max ~ 3l/4 ± ξ where ξ ~ log(l)

So both eigenvalues scale linearly with depth, but the condition number grows with n. That's why deeper networks improve conditioning but with diminishing returns.

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

### Practical Implementation

Now, how do we actually implement this? We have two integral formulations - one with uniform Lebesgue measure, another with the sampling measure from our dataset. The discrete implementation becomes:
L_s[f] ≈ f^T P_s f

The computational complexity is challenging though. We need O(n² ℓ_max^(d-1)) for the matrix construction, where the dimension growth N(d,ℓ) ~ ℓ^(d-2) determines the complexity.

### Deep NTK Analysis

For deep networks, we have this eigenvalue decay μ_k ~ C(d,L)k^(-d) where C(d,L) grows quadratically with L. But there's a really nice result about inverse cosine distance matrices - the NTK matrix has this near-affine behavior: K^∞ ≈ A·W_l + B.

This relationship enables indirect analysis of NTK spectral properties through simpler geometric matrices, which is computationally much more tractable.

### Deep Narrow Networks

This is a fascinating direction. For deep narrow networks, the scaled NTK converges to a Gaussian process limit. The comparison with two-layer kernels is striking:
- Two-layer: μ_ℓ^(2) ~ ℓ^(-(d+1))
- Deep narrow: μ_ℓ^(dn) ~ C(L,d)ℓ^(-d) where C(L,d) ∝ L

So depth partially compensates for the poor conditioning of two-layer kernels. The spectrum is flatter: κ(K^(dn)) ≈ κ(K^(2))/L.

### Alternative Domains

This is where I think there's a lot of potential for future work. We've looked at three domains:

**Gaussian domain**: Uses Ornstein-Uhlenbeck operator with Hermite polynomial eigenfunctions. The Gaussian measure introduces natural regularization through exponential decay.

**Toroidal domain**: This is really promising! Functions on [0,1]^d extended periodically, with Fourier modes as eigenbasis. The key advantage is exact orthogonality of Fourier modes under uniform sampling.

The computational advantage of the torus is decisive. While spherical harmonics aren't orthogonal under uniform sampling, Fourier modes satisfy perfect orthogonality:
(1/N) Σ e^(2πik₁j/N) e^(-2πik₂j/N) = δ_{k₁,k₂ mod N}

This means we can explicitly construct all eigenvectors and use FFT for O(N^d log N) eigenvalue computation!

### Unified Framework

The reconciliation between geometric and functional views comes through shared algebraic structure. Both K and P_s are rotationally invariant, so they commute and are simultaneously diagonalizable. The eigenvalues of the composite operator are simply products:
λ_i(KP_s) = λ_i(K) · λ_i(P_s)

I've also shown that P_s has a zonal kernel representation with explicit Gegenbauer polynomial expansions, which enables fast multipole methods.

### Future Directions

So where do we go from here? The near-term objectives are:
1. Study narrow NTK behavior to identify simplifications
2. Incorporate Sobolev framework into spherical harmonic analysis with experimental validation
3. Unify initialization schemes across different theoretical frameworks

Long-term, we want to extend harmonic analysis to general domains, develop complete theory for deep narrow networks, and create a unified spectral theory encompassing all major NTK variants.

The key insight from Sobolev perspective is that we can tune the exponent s as a function of data size n. When n grows, κ(K) ~ n deteriorates, but multiplying by P_s amplifies high-frequency modes and can flatten the composite spectrum. Making s larger for large n counterbalances K's conditioning.

### Computational Implementation Strategy

For the toroidal domain, the implementation strategy leverages Kronecker structure. The d-dimensional DFT decomposes as F_d = F₁ ⊗ F₁ ⊗ ... ⊗ F₁. The algorithm becomes:
1. Forward FFT: f̂ = F_d f using d successive 1D FFTs
2. Spectral multiplication: ĝ_k = (1 + ||k||²)^s f̂_k  
3. Inverse FFT: g = F_d* ĝ using d successive 1D inverse FFTs

This goes from O(N^(2d)) naive approach to O(d·N^d log N) with FFT, which is a massive improvement.

The practical advantages are clear: we can exploit optimized NumPy/SciPy FFT implementations, use vectorized operations for spectral multiplications, and provide both exact and approximate variants depending on precision needs.

This framework demonstrates the practical advantages of the toroidal approach for large-scale NTK-Sobolev analysis, and I think this is where the most promising experimental work lies for the immediate future.

That's the synthesis of what I've learned - the convergence of spectral analysis, harmonic analysis, and neural network theory really opens up exciting possibilities for understanding and controlling learning dynamics through spectral properties.
