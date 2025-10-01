full report plan outline  : 

it will be in 3 sections 


1st section NTK and sobolev training:
- we state our problem in 1/2 page (this has been partially done), that sobolev training has 2 branches, and we focus only in the 1st branch in the purpose of NN regression for FNO
- we state def for NTK in 1 page, sobolev matrix/operator in 1 page
- then we do a litterature review in 1 page
- then we disentangle the NTK matrix & operator, same for sobolev matrix and operator, in 1 page, methodology and personnal contribution part
- then we state the main theorem for commutations (this is the results), proofs in appendix, personal contribution
- then we discuss the results : the main result is that we disentangle the dimensionnality of our data in the sobolev matrix, and the network part in the NTK (depth and width), personnal contribution
- in this way the rkhs is disentangled in 2 parts (state only the results for the final spectrum, that is the multiplication of each)
- we derive then an intuitive agenda to undersand each contribution : weyl inequality for ntk matrix, and dimensionnality analysis for P matrix, personnal contribution
- make a blank part where we will discuss the 2nd branch, where we have derivative data (for pinn or pinn deeponets)

2nd section MMNNS: 
- we state our problem, that for when the pinn loss is used, MMNNs appears to deliver great optimization results etc .., 1 page
- we state def for mmnn, and implementation and results from zhang paper, 1 page
- we do a litterature review in the field of improving optimizations bounds for mmnns, how to get those bounds etc .. (leave this blank), 1 page
- then we state theorems where compute NTK for mmnns, recursive relation, EOC,  3 pages, personal contribution, proofs are put in appendix in their totality, leave blank a part where we state a theorem for optimization bounds of smallest eigenvalue
- then we discuss the results, the scaling, the NTK randomness for MMNNs, the std, how it behave and how it can become fairly well better than FCNN, 1 page, personal contribution
- then we discuss numerical results (as done in the experiments), all plots are in appendix, talk about them by refs, the intuition etc .; 2 page, personal contribution
- in this way we can also derive the RKHS with bach/bietti results, leave this blank, 1 page



4th section : FCNN & transformers

- we introduce the NTK for FCNN and transformers and state the main theorems (just structure of network, some definitions etc ..), and finite width corrections 1 page
- we do a concise but very complete litterature review of ntk results (greg yang, sofiane hayou, seleznova, ethan dyer etc ..)  2 page
- We state all theoremes personal contributions :hessians fisher and NTK spectrum are the same, finite width corrections formulas and approximated scaling wrt depth 2 pages, NTK for transformers and formulas, intractableness
- we discuss those theorems, intuition with weyl inequality, intuition to deal with NTK transformers etc .. 2 pages
- we state numerical results for finite width corrections scaling, from theory and discuss them 2 pages



be careful, the orthogonality under sampling measure is orthogonal in mean ! that is the ntk matrix is random, and that's when we take the 



# plan generated (to write here)

# Detailed Report Outline

This plan is based on the structure defined in the planning files and incorporates elements already present in `main.tex`. The objective is to structure the existing content and guide the writing of the missing parts to form a coherent research report.

---

## Part 1: NTK and Sobolev Training

*   **1.1. Abstract**
    *   **Problem Statement:** Sobolev training for neural networks (NNs) is divided into two approaches. We focus on the first: using Sobolev norms as a regularization for NN regression, particularly in the context of Fourier Neural Operators (FNOs) for solving PDEs.
    *   **Methodology:** We propose a theoretical decomposition of the Neural Tangent Kernel (NTK) and the Sobolev operator. The goal is to separate the influence of the network architecture (width, depth) from that of the data structure (dimensionality, regularity).
    *   **Main Results:** We prove a commutation theorem between the NTK operator and the Sobolev operator. This result implies that the spectrum of the combined kernel is the product of the individual spectra, allowing for a decoupled analysis.
    *   **Main Conclusion:** This decomposition offers a new perspective for analyzing and understanding the performance of FNOs and similar methods, by explicitly linking the network architecture to the functional structure of the data space.

*   **1.2. Introduction**
    *   **Establishing the territory:** "In recent years, research has focused on the theoretical understanding of deep neural networks, particularly through the infinite-width regime and the NTK formalism." Introduce the growing importance of learning methods for scientific problems (SciML) and solving PDEs.
    *   **Establishing a niche:** "However, the interaction between the learning dynamics described by the NTK and the inductive biases imposed by functional regularizations, such as Sobolev training, remains unclear." Highlight the lack of a unified framework.
    *   **Occupying the niche:** "This work aims to fill this gap by providing a rigorous spectral analysis of the NTK for networks trained with a Sobolev norm. We present a decomposition of the operators, discuss its implications, and propose an agenda for analyzing the components." Announce the structure of the part.

*   **1.3. Literature Review**
    *   **NTK Theory:** Review of foundational works (Jacot et al.), "kernel" and "feature learning" regimes, scaling laws.
    *   **Sobolev Training:** Review of existing methods, applications in physics (PINNs), and for neural operators.
    *   **Neural Operators:** FNOs, DeepONets, and their connection to Sobolev spaces.

*   **1.4. Methodology and Personal Contributions**
    *   **Formal Definitions:**
        *   NTK: Definition for FCNNs, recurrence formula.
        *   Associated Reproducing Kernel Hilbert Space (RKHS).
        *   Sobolev Operator and Matrix.
    *   **Decomposition of Operators:**
        *   Clarify the distinction between the NTK matrix (on the data) and the NTK operator (in the function space).
        *   Same for the Sobolev operator/matrix.
        *   Present the intuition of the separation: the architecture's contribution is captured by the NTK, the data's by Sobolev. *(Material to be extracted from `main.tex`, section "Reconciling NTK matrix spectrum and Sobolev Training")*

*   **1.5. Results**
    *   **Main Theorem: Commutation of Operators**
        *   Formal statement of the theorem.
        *   Proof sketch, highlighting the key steps. (Full proof in Appendix).
    *   **Corollary: Spectrum of the Combined Kernel**
        *   The final spectrum is the tensor product of the spectra.
        *   Explicit formulation of the resulting RKHS.

*   **1.6. Discussion**
    *   **Interpretation:** The main result allows for a modular analysis. One can separately study the effect of depth/width (via the NTK spectrum) and the effect of data dimension/regularity (via the Sobolev operator spectrum).
    *   **Agenda for Spectral Analysis:**
        *   Use Weyl's inequality to bound the NTK eigenvalues.
        *   Analysis of the decay of the Sobolev operator's eigenvalues as a function of dimension.
    *   **Limitations and Future Work:**
        *   Mention the second branch of Sobolev training (using derivative data for PINNs) as a future direction.

---

## Part 2: Multi-scale Matrix-multiplication Neural Networks (MMNNs)

*   **2.1. Abstract**
    *   **Problem Statement:** MMNNs have shown remarkable empirical performance for multi-scale problems (e.g., PINNs), but their theoretical analysis remains embryonic.
    *   **Methodology:** We derive for the first time the NTK formulas for MMNNs, establishing a recurrence relation and analyzing signal propagation (Edge of Chaos).
    *   **Main Results:** Explicit formulas for the NTK, analysis of its mean, variance, and scaling. We show properties that favorably distinguish it from the FCNN's NTK.
    *   **Main Conclusion:** The NTK analysis of MMNNs provides a theoretical explanation for their good optimization performance and paves the way for a better understanding of their inductive bias.

*   **2.2. Introduction**
    *   **Establishing the territory:** Present MMNNs as an architecture suited for multi-scale problems in SciML.
    *   **Establishing a niche:** "Despite their empirical success, the training dynamics and optimization landscape of MMNNs are poorly understood."
    *   **Occupying the niche:** "We propose a theoretical analysis of MMNNs via the NTK formalism. We derive its expression, study its spectral properties, and validate our results with numerical experiments."

*   **2.3. Literature Review**
    *   **MMNNs:** Original paper by Zhang et al., applications, and variants.
    *   **Optimization bounds via NTK:** Review of results linking the smallest eigenvalue of the NTK to convergence speed.

*   **2.4. Theoretical Results (Personal Contributions)**
    *   **Definitions:** MMNN architecture, asymptotic regime.
    *   **Signal Propagation and Edge of Chaos (EOC):** *(Material to be extracted from `main.tex`, sections "Signal Propagation and the Edge of Chaos (EOC)")*
    *   **Theorems on the NTK of MMNNs:**
        *   Calculation of the NTK for a single layer. *(Material in "NTK for a Single-Layer MMFN")*
        *   Recurrence formula for multiple layers. *(Material in "Recursion for Multi-Layer MMNNs")*
        *   Variance analysis (EOC).
    *   (Section to be completed) Theorem on the bounds of the smallest eigenvalue.
    *   Full proofs are relegated to the Appendix.

*   **2.5. Numerical Results and Discussion**
    *   **Analysis of the MMNN's NTK:**
        *   Comparison of scaling (depth/width) with FCNNs.
        *   Discussion on randomness and variance.
    *   **Experimental Validation:**
        *   Present numerical experiments validating the theoretical scaling laws. *(Material in "Résultats Expérimentaux et Applications", "Expériences numériques")*
        *   Discuss the intuition behind the observed results (reference to figures in the Appendix).
    *   (Section to be completed) Link with the RKHS via the results of Bach/Bietti.

*   **2.6. Conclusion and Outlook**
    *   **Summary of contributions.**
    *   **Limitations:** Difficulty in obtaining global optimization bounds.
    *   **Conjecture and future work:** "We conjecture that the optimization landscape is well-conditioned around global minima, as our experiments suggest." Avenues for proving this conjecture.

---

## Part 3: FCNNs, Finite Width, and Transformers

*   **3.1. Abstract**
    *   **Problem Statement:** NTK theory is primarily asymptotic. Understanding the impact of finite width is crucial for practical applications. Furthermore, the NTK of Transformers remains difficult to characterize.
    *   **Methodology:** We develop finite-width correction formulas for the FCNN's NTK using tools from random matrix theory (RMT). We also compute the NTK for a simplified Transformer-like architecture.
    *   **Main Results:** Explicit formulas for finite-width corrections and their scaling. Formula for the Transformer's NTK, highlighting its complexity and its connection to higher-order tensors.
    *   **Main Conclusion:** Our results refine NTK theory for the finite-width regime and shed light on the theoretical challenges posed by attention-based architectures.

*   **3.2. Introduction**
    *   **Establishing the territory:** The success of the infinite-width NTK.
    *   **Establishing a niche:** "However, practical networks have finite width, and non-asymptotic corrections are essential. Moreover, for complex architectures like Transformers, a complete NTK theory is lacking."
    *   **Occupying the niche:** "We address both points. First, we present finite-width corrections for FCNNs. Second, we derive the NTK for Transformers and discuss its intractability."

*   **3.3. Literature Review**
    *   **Results on finite-width NTK:** Review of works by Greg Yang, Sofiane Hayou, Selesnova, Ethan Dyer, etc.
    *   **Random Matrix Theory (RMT) applied to NNs.**
    *   **NTK for Transformers.**

*   **3.4. Theoretical Results (Personal Contributions)**
    *   **Spectral Equivalence:**
        *   Theorem: The spectra of the Hessian, the Fisher Information Matrix, and the NTK coincide in the infinite-width limit. *(Material in "The Hessian-NTK Correspondence: A Spectral Equivalence")*
    *   **Finite-Width Corrections for FCNNs:**
        *   Theorem: Correction formulas and approximate scaling with depth. *(Material in "Corrections à Largeur Finie et Régimes d'Apprentissage")*
        *   Link with RMT (GOE/GUE). *(Material in "A Random Matrix Perspective on the NTK Correction", "The Gaussian Orthogonal Ensemble in the NTK Spectrum")*
    *   **NTK for Transformers:**
        *   Theorem: Calculation of the NTK for a simplified architecture, highlighting the involvement of 4th-order tensors ($O_4$). *(Material in "On the Fourth-Order Kernel $O_4$ and its Computation")*

*   **3.5. Discussion and Numerical Results**
    *   **Intuition and Interpretation:**
        *   Use of Weyl's inequality to interpret the impact of the corrections.
        *   Discussion on the computational challenges of the Transformer's NTK.
    *   **Numerical Validation:**
        *   Present experiments validating the scaling laws of the finite-width corrections. *(Material in "Experimental Validation")*
        *   Compare theory and practice.
    *   **Discussion on the last term (QUE & GOE):** Final intuition, results, and discussions.

*   **3.6. Conclusion**
    *   **Summary of contributions** on FCNNs and Transformers.
    *   **Future work:** Avenues for a more in-depth analysis of Transformers.

---

## Appendix and Numerical Results: Detailed Structure

This section provides a detailed breakdown of the content for the appendices. The goal is to separate the main narrative of the report from the extensive proofs, raw data, and supplementary material, ensuring clarity and readability.

### Appendix A: Proofs of Main Theorems

*   **A.1 Proofs for Part 1: NTK and Sobolev Training**
    *   **A.1.1 Full Proof of the Commutation Theorem:** Detailed step-by-step derivation of the commutation property between the NTK and Sobolev operators.
    *   **A.1.2 Derivation of the Combined RKHS:** Explicit construction and characterization of the resulting Reproducing Kernel Hilbert Space.

*   **A.2 Proofs for Part 2: MMNNs**
    *   **A.2.1 Proof of the NTK Recurrence Relation:** Derivation of the layer-by-layer recurrence formula for the MMNN NTK. *(Material from "Main Proofs", "Proof of the Recursive Formulas")*
    *   **A.2.2 Derivation of Signal Propagation at EOC:** Detailed derivation for the recurrence of the variance of activations, proving the conditions for maintaining signal propagation (Edge of Chaos). *(Material from "Appendix: Derivation of the Variance Recurrence (EOC)")*

*   **A.3 Proofs for Part 3: FCNNs, Finite Width, and Transformers**
    *   **A.3.1 Proof of Spectral Equivalence (Hessian-Fisher-NTK):** Rigorous proof showing the asymptotic equivalence of the spectra for the Hessian, Fisher Information Matrix, and the NTK. *(Material from "The Hessian-NTK Correspondence: A Spectral Equivalence")*
    *   **A.3.2 Derivation of Finite-Width Correction Formulas:** Step-by-step derivation of the first-order correction terms for the NTK.
    *   **A.3.3 Derivation of the NTK for Transformers:** Full calculation for the simplified attention mechanism, including the explicit computation of the fourth-order tensor $O_4$. *(Material from "On the Fourth-Order Kernel $O_4$ and its Computation")*

### Appendix B: Numerical Results and Experimental Validation

*   **B.1 Experimental Setup**
    *   **B.1.1 Implementation Details:** Description of the computational environment: hardware (CPU/GPU), software libraries (e.g., PyTorch, JAX), and versions.
    *   **B.1.2 Code and Reproducibility:** Link to a public code repository (e.g., GitHub). Notes on how to run the experiments to reproduce the figures and results. A section mapping key mathematical formulas to the corresponding code functions. *(Material from "Implementation of the NTK, and the code correspondence")*

*   **B.2 Experimental Results for Part 2: MMNNs**
    *   **Plots:**
        *   *Figure B.1:* Comparison of the NTK eigenvalue spectrum for MMNNs vs. FCNNs as a function of network depth.
        *   *Figure B.2:* Comparison of the NTK eigenvalue spectrum for MMNNs vs. FCNNs as a function of network width.
        *   *Figure B.3:* Empirical validation of the Edge of Chaos condition, plotting activation variance across layers.
        *   *Figure B.4:* Training loss curves for MMNNs and FCNNs on a benchmark task, demonstrating faster convergence for MMNNs.
    *   **Data Tables:** Tables with raw data corresponding to the plots for detailed inspection. *(Material from "Résultats Expérimentaux et Applications", "Expériences numériques")*

*   **B.3 Experimental Results for Part 3: Finite-Width Corrections**
    *   **Plots:**
        *   *Figure B.5:* Empirical vs. Theoretical scaling of the largest NTK eigenvalue with width ($N$). The plot should show the raw empirical data against the theoretical curve from the derived formulas.
        *   *Figure B.6:* Empirical vs. Theoretical scaling of the NTK trace with width ($N$).
        *   *Figure B.7:* Histogram of the empirical NTK eigenvalues for a large-width network, overlaid with the predicted spectral density from Random Matrix Theory (GOE/GUE).
    *   **Data Tables:** Tables containing the raw scaling data for width ($N$) and depth ($L$). *(Material from "Appendix A: Raw Scaling Data for N", "Appendix B: Raw Scaling Data for L")*

### Appendix C: Complementary Theoretical Derivations

*   **C.1 Price's Theorem and Applications:** A self-contained section detailing Price's Theorem and showing its application in the context of computing expectations of correlated Gaussian variables in the NTK derivations. *(Material from "Price's Theorem and Derivations")*
*   **C.2 Review of the Tensor Programs (TP) Framework:** A brief tutorial on the Tensor Programs framework, explaining the notation and key results used in the literature review and for some derivations.

### Appendix D: Project Development and Contributions

*   **D.1 History of Research Directions:** A chronological summary of the project's evolution, including a graph illustrating the different research paths explored over time.
*   **D.2 Engineering and Implementation Contribution:** Details on the experimental setup, including commit statistics, lines of code written, and a description of the custom codebase developed for this research.