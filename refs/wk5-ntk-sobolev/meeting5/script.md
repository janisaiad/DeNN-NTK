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

Now, why is this analysis important? The primary objective is to establish optimization bounds under the NTK perspective for neural network regression and solving partial differential equations. This complements the existing theoretical framework in several crucial ways.

Recent work, notably by Yang & He, has already established robust generalization and approximation bounds for deep super-ReLU networks. Their comprehensive analysis demonstrates that deep networks can achieve optimal generalization rates under Sobolev loss while simultaneously enabling efficient approximation of functions in Sobolev spaces through architectural depth. These foundational results establish that the theoretical framework for understanding both statistical learning guarantees and representational capacity is well-developed.

However, an important gap remains: optimization bounds in the NTK regime are largely unexplored. While we understand how well these networks can generalize and what functions they can represent, the fundamental question of how efficiently they can be trained remains unanswered from a rigorous theoretical perspective.

Our spectral analysis aims to fill this gap by establishing a comprehensive understanding of convergence rates through the lens of NTK spectral properties, revealing how these properties determine optimization convergence rates in practice. Additionally, we seek to characterize the intricate relationships between network depth, data size, and optimization difficulty through matrix conditioning analysis. Finally, our framework elucidates how Sobolev training modifies the underlying spectrum to favor certain frequency components, providing practitioners with principled approaches to control learning dynamics.

This NTK perspective proves particularly relevant for PDE applications, where fine control of Fourier components directly determines the quality of the numerical solution, making the intersection of spectral analysis and optimization theory essential for advancing computational mathematics.

### Optimization Bounds via Spectral Factorization

A fundamental advantage of our approach lies in the factorization of the Sobolev-modified learning operator as T_s = K^∞ ∘ P_s. This decomposition allows us to disentangle the effects of the neural network architecture, encoded in the NTK matrix K^∞, from the regularization effects of Sobolev training, encoded in the operator P_s.

By leveraging the commutation property [K^∞, P_s] = 0 that arises from rotational invariance, we can study each component separately and then combine their spectral properties through simple multiplication of eigenvalues. This factorization strategy enables us to focus specifically on deriving optimization bounds for the NTK matrix alone, characterizing how the spectral properties depend on architectural parameters such as network depth L, width N, and more sophisticated architectural choices like parabolic width profiles m_ℓ = m^(ℓ²) for MLPs at the edge of chaos.

### Perturbation Theory Framework for Finite-Width Effects

To achieve these optimization bounds, we employ a systematic perturbation theory approach that treats finite-width effects as corrections to the infinite-width limit. Our starting point is the infinite-width regime where N → ∞, in which the neural network behavior is completely characterized by the NTK operator. The spectrum of this infinite-width NTK serves as our "theoretical unperturbed spectrum" H₀, representing the ideal baseline for optimization analysis.

The key insight is that finite width N introduces systematic corrections to this ideal behavior. We can formulate the finite-width system as H = H₀ + V, where H₀ represents the infinite NTK operator and V captures the perturbation due to finite width, typically scaling as O(1/N). This perturbative framework allows us to systematically compute how architectural choices affect the spectrum and, consequently, optimization convergence rates.

The calculation of these perturbative corrections requires advanced tools from modern neural network theory, particularly the NTK hierarchy and Feynman diagram techniques developed by researchers like Yaida, as well as the Tensor Programs framework introduced by Yang. These methods, inspired by quantum field theory, provide systematic approaches to compute higher-order corrections beyond the standard NTK approximation.

This perturbative approach naturally addresses the fundamental "deeper versus wider" question in neural network design. By fixing a total parameter budget P and expressing the finite-width corrections in terms of both depth L and effective width N = P/L, we can analytically determine whether increasing depth or width provides better optimization guarantees. The spectral analysis reveals how the smallest eigenvalue λ_min, which controls convergence rates, responds to these architectural trade-offs under the constraint of fixed computational resources.

### Strategic Component-wise Investigation

Our factorization approach T_s = K^∞ ∘ P_s reveals an important strategic advantage that significantly simplifies the research program. Since we have successfully disentangled the Sobolev operator P_s from the neural network kernel K^∞, we can now focus on investigating each component independently before combining their effects. This separation transforms an inherently complex joint optimization problem into two more manageable subproblems, each amenable to specialized mathematical techniques.

For the Sobolev operator P_s, the investigation can proceed through well-designed experimental frameworks that leverage Fourier analysis on various domains. These experiments can systematically explore how different values of the Sobolev parameter s modify the spectral structure, providing empirical validation of our theoretical predictions about frequency bias and learning dynamics.

The investigation of the NTK matrix K^∞ under finite-width corrections represents genuinely new territory that extends beyond the current state of knowledge. While existing literature has established important results about bounding the smallest eigenvalue through width scaling laws, these works fundamentally differ from our proposed approach. Our research path represents a conceptually different approach that explicitly assumes NTK regime validity and then investigates how finite-width corrections affect optimization bounds within this constrained setting.

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

### Finite-Width Corrections and NTK Hierarchy

Now, let me discuss the advanced mathematical machinery needed for optimization bounds. The finite-width NTK expansion takes this explicit form:

Θ^NTH(x₁, x₂) = Θ(x₁, x₂) + n^(-1)𝔼[𝒪^(1)_{2,0}(x₁, x₂)] - n^(-1)𝔼[𝒪^(1)_{3,0}(x₁, x₂, x⃗)Θ^(-1)(x⃗, x⃗)f^(0)_0(x⃗)] + n^(-1)y⃗^T Θ^(-1)(x⃗, x⃗)𝔼[𝒪^(1)_{4,0}(x₁, x₂, x⃗, x⃗)] Θ^(-1)(x⃗, x⃗)y⃗ + higher order terms

These corrections capture how the idealized infinite-width behavior degrades in practice. The tensors 𝒪^(1)_{4,0} and 𝒪^(1)_{3,0} represent higher-order interaction terms that require detailed investigation to understand their impact on spectral properties and optimization dynamics.

This represents a conceptually different approach from existing literature. While previous work has established bounds on λ_min through width scaling laws, these approaches don't assume operation within the Neural Tangent Kernel regime. Instead, they utilize the NTK definition as a mathematical tool while conducting extensive linear algebraic analysis without the crucial assumption that the network actually behaves according to NTK dynamics during training.

Our approach explicitly assumes NTK regime validity and then investigates how finite-width corrections affect optimization bounds within this constrained setting. This opens new theoretical questions about the interplay between kernel constancy, spectral properties, and architectural parameters within the lazy training regime.

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

### Future Directions and Research Roadmap

So where do we go from here? Our approach opens several exciting research directions that build on the optimization bounds framework.

**Near-term objectives** focus on developing the perturbation theory foundation:
1. Study narrow NTK behavior to identify potential simplifications before developing more general approaches
2. Integrate the Sobolev framework with spherical harmonic analysis and validate our findings experimentally  
3. Work toward unifying initialization schemes and architectural assumptions across different theoretical frameworks to build a stronger foundation

**Finite-width corrections investigation** represents two complementary paths. The first involves investigating finite-width corrections, which represent a mathematically challenging but theoretically rigorous approach. The finite-width NTK expansion includes complex expressions with multiple interaction terms scaling as n^(-1), requiring sophisticated mathematical machinery including tensor algebra and careful treatment of statistical dependencies between network parameters.

The second path focuses on the asymptotics of wide networks through Feynman diagram techniques, leveraging tools from quantum field theory to systematically compute higher-order corrections to the NTK, treating neural network training as a many-body interacting system where neurons exhibit complex collective behavior beyond the mean-field approximation.

**Long-term goals** envision extending harmonic analysis from spherical to general domains while maintaining strong experimental validation. We aim to develop a complete theory for deep narrow networks with clear practical applications, and ultimately create a unified spectral theory that encompasses all major NTK variants and training modifications.

The key insight from Sobolev perspective is that we can tune the exponent s as a function of data size n. When n grows, κ(K) ~ n deteriorates, but multiplying by P_s amplifies high-frequency modes and can flatten the composite spectrum. Making s larger for large n counterbalances K's conditioning. However, we still lack a precise estimate of κ(P_s) - an important problem that warrants future investigation.

**Novel territory focus**: The investigation of NTK analysis within the lazy training regime represents genuinely new territory. While existing bounds are derived through direct matrix analysis without requiring the kernel to remain approximately constant throughout training, our approach explicitly assumes NTK regime validity and investigates how finite-width corrections affect optimization bounds within this constrained setting. This perspective shift opens new theoretical questions about the interplay between kernel constancy, spectral properties, and architectural parameters.

### Computational Implementation Strategy

For the toroidal domain, the implementation strategy leverages Kronecker structure. The d-dimensional DFT decomposes as F_d = F₁ ⊗ F₁ ⊗ ... ⊗ F₁. The algorithm becomes:
1. Forward FFT: f̂ = F_d f using d successive 1D FFTs
2. Spectral multiplication: ĝ_k = (1 + ||k||²)^s f̂_k  
3. Inverse FFT: g = F_d* ĝ using d successive 1D inverse FFTs

This goes from O(N^(2d)) naive approach to O(d·N^d log N) with FFT, which is a massive improvement.

The practical advantages are clear: we can exploit optimized NumPy/SciPy FFT implementations, use vectorized operations for spectral multiplications, and provide both exact and approximate variants depending on precision needs.

This framework demonstrates the practical advantages of the toroidal approach for large-scale NTK-Sobolev analysis, and I think this is where the most promising experimental work lies for the immediate future.

That's the synthesis of what I've learned - the convergence of spectral analysis, harmonic analysis, and neural network theory really opens up exciting possibilities for understanding and controlling learning dynamics through spectral properties.

### Key Takeaways and Impact

The main impact of this work is that we've established a systematic framework for understanding optimization bounds in the NTK regime. The factorization T_s = K^∞ ∘ P_s allows us to separate architectural effects from regularization effects, making the analysis much more tractable.

Our perturbation theory approach provides a principled way to understand how finite-width effects degrade optimization performance. This is crucial because real networks always have finite width, and the infinite-width analysis is just our starting point.

The deeper-versus-wider question can now be addressed analytically by fixing a parameter budget P and comparing how depth L versus width N = P/L affects the smallest eigenvalue λ_min, which controls convergence rates.

For PDE applications, this framework is particularly valuable because we can now predict and control which frequency components will be learned effectively by tuning the Sobolev parameter s. This level of control over the learning dynamics is exactly what we need for high-quality numerical solutions.

The experimental validation on the toroidal domain provides a concrete path forward for testing these theoretical predictions at scale, with computational advantages that make large-scale studies feasible.

This represents a significant step toward completing the theoretical picture of neural networks - we now have generalization bounds, approximation bounds, and a clear path toward optimization bounds, all within a unified spectral framework.
