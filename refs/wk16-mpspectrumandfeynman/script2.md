(Slide 1: Title) "Good morning. Today, I'll be presenting the work by Guillen, Misof, and Gerken, which introduces a systematic diagrammatic framework for computing finite-width NTK corrections."

(Slide 2: The Problem: Beyond Infinite Width) "As this audience is well aware, the standard infinite-width NTK theory provides a powerful but limited picture. The linearization of the model and the freezing of the kernel preclude the analysis of feature learning. The standard approach to move beyond this is, of course, a perturbative expansion in the inverse layer width, 
1
/
n
1/n. The challenge has always been the combinatorial complexity of computing the coefficients of this series, especially for correlators involving the NTK itself. This paper's main contribution is to provide a systematic solution to this computational bottleneck."

(Slide 3: The Formalism: Joint Cumulants) "The authors make the correct theoretical choice of framing the problem in terms of joint cumulants, or connected correlators, rather than raw moments. This is the natural language for interactions in a perturbative expansion, as cumulants isolate the non-Gaussian statistics that are identically zero at infinite width. They then introduce a tensor basis—the familiar A, B, D, F tensors and their generalizations—to decompose these cumulants. The goal then becomes to find the recursion relations for these basis tensors."

(Slide 4: The Tool: Feynman Diagrams for NTKs) "The central innovation is porting the machinery of Feynman diagrams from quantum field theory to this context. The rules are analogous: propagators, shown here as the white blob, represent the Gaussian part of the statistics—that is, the covariance from the previous layer. Vertices represent the non-Gaussian interactions introduced by the current layer's nonlinearity. The expansion parameter is 
1
/
n
1/n. This framework is powerful because it provides a graphical representation of the underlying algebraic structure, allowing for an exhaustive generation of all contributions at a given order."

(Slide 5: Application 1: Deriving Recursion Relations) "The power of this formalism becomes immediately apparent when deriving recursion relations. To find the F-tensor at layer 
ℓ
+
1
ℓ+1, for example, one simply draws all connected diagrams of order 
1
/
n
1/n with the appropriate external lines. As shown, there are only two such topologies. Each diagram then maps directly to an algebraic term via the Feynman rules. This automates what was previously an extremely tedious and error-prone algebraic exercise."

(Slide 6: Application 2: NTK Mean Correction) "Armed with this tool, the authors derive what is, to my knowledge, the first complete recursion relation for the leading-order, 
1
/
n
1/n correction to the NTK mean. This quantity is of central importance, as it describes the leading-order deviation of the average training dynamics from the frozen infinite-width picture. The five diagrams shown here represent the full contribution at this order. The complexity of this expansion makes it obvious why this was intractable without such a systematic approach."

(Slide 7: Application 3: Stability and Scale-Invariance) "Now, let's discuss two of the most powerful applications of the formalism, which directly address the scaling with network depth.

First, the stability argument. The diagrammatic structure allows for an elegant all-orders proof of gradient stability. The key insight is that the layer-to-layer propagation of any tensor's variation is governed by a susceptibility factor, 
χ
χ. Imposing the infinite-width criticality condition sets this susceptibility to one, which prevents the exponential explosion that would render the theory useless for deep networks.

But the crucial question is: if it's not exponential, how do finite-width corrections scale with depth, 
ℓ
ℓ? The growth is not zero; it is polynomial. Their numerical experiments in Appendix H are very revealing. They measure the scaling exponents for various tensor components at criticality. For the V4 preactivation cumulant, they find that it scales with depth as 
ℓ
1.2
ℓ 
1.2
  to 
ℓ
1.5
ℓ 
1.5
 . For the NTK-related tensors, the growth is faster: the D and F tensors scale roughly as 
ℓ
2.2
ℓ 
2.2
  to 
ℓ
2.7
ℓ 
2.7
 , while the NTK variance tensors A and B exhibit the fastest growth, scaling as 
ℓ
3.2
ℓ 
3.2
  to 
ℓ
3.8
ℓ 
3.8
 . The fact that these exponents are non-integer and tensor-dependent is a highly non-trivial prediction, showcasing the rich structure of finite-width effects even in the stable regime.

Second, the ReLU cancellation. This result is remarkable. The formalism proves that for scale-invariant activations, the finite-width correction to the NTK diagonal, 
Θ
(
x
,
x
)
Θ(x,x), is identically zero. This is an exact cancellation, not an approximation, stemming from profound algebraic identities like the one shown. This proves that for this specific observable, the infinite-width theory is exact, a fact previously known but now demonstrable within a general framework."

(Slide 8: Conclusion) "In conclusion, Guillen, Misof, and Gerken have provided a significant contribution to the field of finite-width corrections. They've developed a principled, systematic, and extensible diagrammatic framework that tames the combinatorial complexity of the 
1
/
n
1/n expansion. This not only makes existing calculations trivial but also enables new results like the NTK mean correction and powerful all-orders stability proofs with concrete polynomial scaling predictions. The clear path forward is to use this machinery to compute the next-to-leading order (
1
/
n
2
1/n 
2
 ) corrections and to begin adapting the rules for more complex architectures like CNNs and Transformers. Thank you."
