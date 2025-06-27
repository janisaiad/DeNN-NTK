"CONCISE"














"LONG"

Okay, so for today's presentation, I'm going to walk you through the work I've been doing. First, a quick personal update: I've had to spend a lot of time sorting out administrative issues with the embassy and the French consulate, which has been quite a saga. However, this gave me a lot of time to have experiments running in the background to really nail down the scaling laws I've been observing. It also gave me the chance to think more deeply about the theoretical side, specifically from the perspective of the NTK's spectral properties and Random Matrix Theory.

---

### The Presentation Script

**(Slide 1: Title)**

Alright, so today I'm presenting my latest results on understanding the finite-width NTK correction, framed through a Random Matrix Theory perspective.

**(Slide 2: Outline)**

Here's the plan. We'll start with the main goal, which is to get the scaling laws for the NTK correction. Then, I'll walk you through a detailed scaling analysis of the different terms involved. This analysis will reveal a puzzle when we compare the raw theory to my experimental results, and I'll discuss how we can reconcile that. Finally, I'll introduce a more formal RMT framework which provides a much deeper intuition for what's happening, especially about the role of network depth in learning.

**(Slide 3: The Late-Time NTK Correction)**

The main object we are trying to understand is this: $\Theta^{(1)}_\infty$, the leading-order correction to the NTK at the end of training. It tells us how much the kernel has changed from its initial state. The formula from the literature splits it into two main components, which I've labeled $T_3$ and $T_4$. $T_3$ depends on the third-order kernel $O_3$, and $T_4$ on the fourth-order kernel $O_4$. To understand the whole correction, we have to understand how these two big pieces scale with network and data parameters, like depth $L$ and dataset size $N$.

**(Slide 4: The NTK Spectrum: A Dichotomy)**

The key to analyzing these sums is understanding the spectral structure of the initial NTK. Empirically, and this is confirmed in my experiments, we see a clear dichotomy. There's one very large, isolated eigenvalue, $\lambda_1$. Its eigenvector corresponds to the constant mode—it essentially captures the average bias of the function. This eigenvalue scales linearly with depth, so $\lambda_1 \sim \mathcal{O}(L)$.

The other $N-1$ eigenvalues are all clustered together in what we call a "bulk." They are much smaller and all scale similarly, like $\mathcal{O}(L/N)$. Our working hypothesis is that their corresponding eigenvectors behave like random vectors, uniformly distributed on the sphere. This split is the foundation of the whole analysis.

**(Slide 5: Scaling Analysis of the $O_3$ Term)**

So, let's apply this split to the $T_3$ term. We separate the sum into the contribution from the constant mode ($i=1$) and the contribution from the bulk ($i \ge 2$). When we plug in the scaling for the eigenvalues, we see the bulk term has a pre-factor of $N/L$. Since there are $N-1$ terms in that sum, the whole expression is dominated by the bulk part and scales like $\mathcal{O}(N^2/L)$. This is a very fast growth in $N$.

**(Slide 6: Scaling Analysis of the $O_4$ Term)**

We do the exact same thing for the $T_4$ term. It's a double sum, so there are a few more interactions, but the dominant part is the bulk-bulk interaction, where both indices $i$ and $j$ are greater than 1. Here, the denominator contains a product of two eigenvalues, so it scales like $(L/N)^2$. The sum itself contains roughly $N^2$ terms. Multiplying the number of terms by the pre-factor gives us an overall scaling of $\mathcal{O}(N^4/L^2)$. This is an even more explosive growth with the dataset size $N$.

**(Slide 7: Reconciling Theory and Experiment)**

This leads us to a major puzzle. This raw theoretical analysis suggests the correction should blow up with dataset size, scaling like $N^2$ or even $N^4$. But my extensive experiments show a much more controlled, almost linear scaling, closer to $\mathcal{O}(N)$.

So, what's the missing piece? The only way to reconcile these two is if the tensors $O_3$ and $O_4$, when projected onto these random bulk eigenvectors, have an average magnitude that decays with $N$. For the $T_4$ term, which is the most aggressive, to align with experiments, its expected value would need to decay very quickly, something like $\mathcal{O}(1/N^3)$. This is a new, crucial hypothesis that we need to investigate.

And this is where the real work begins. It's not enough to just do what physicists often do: make a log-log plot, see a straight line, and declare a scaling law. The goal is to *interpret* that law in the context of computer science and the big questions we have today, like the ones raised by scaling law research from places like OpenAI. We want to use this theory to understand the fundamental trade-offs between depth $L$, width $M$, and data size $N$. For a given parameter budget, is it better to build deeper or wider? Answering this puzzle is the first step toward building a theory that can guide architectural choices.

**(Slide 8: Deeper Dive: RMT Framework)**

To formalize this whole discussion about random eigenvectors and spectral distributions, we can use the language of Random Matrix Theory. And I want to emphasize how novel this angle is. Most theoretical work on wide networks uses tools from high-dimensional probability and statistics, but very few people have applied the specific, powerful machinery of RMT to analyze the NTK's *eigenvectors* and *spectral edge dynamics*. There are very few direct references, which meant I had to do a lot of foundational reading to connect the dots between established RMT and our specific problem.

The idea is to view the NTK not as a fixed, deterministic kernel, but as a sample covariance matrix. If the data has no structure, RMT predicts that its eigenvalue spectrum will follow a universal distribution—the Marchenko-Pastur law. This is the "bulk," the baseline structural noise of the network. But when there is structure in the data, it can create "spikes," or outliers in the spectrum. These spikes *are* the learnable features.

**(Slide 9: The BBP Phase Transition and Feature Learning)**

This brings us to the core idea: feature learning *is* a phase transition. A feature, a spike, will only emerge from the random bulk if its signal strength is high enough to cross a critical threshold. This is known as the BBP phase transition. If no data signal is strong enough, the network stays in the "lazy regime." It simply interpolates the data using its pre-existing random structure. But if a signal is strong enough to cross the threshold, the network enters the "feature learning regime." It has identified a meaningful structure, and its representations will actively change to learn it.

**(Slide 10: The Effect of Depth L on Learning)**

So finally, what is the role of depth in this picture? Each layer in a deep network adds its own randomness. From an RMT perspective, the spectrum of the total NTK is the "free convolution" of the spectra of the individual layer kernels. This operation has the effect of widening the Marchenko-Pastur bulk. A wider bulk means its edge moves to the right, which means the BBP threshold gets higher.

And this is the profound consequence of depth: a deeper network requires a *stronger signal* from the data to learn a feature. A feature that a shallow network could easily pick up might be completely drowned out by the structural noise of a much deeper network. This suggests there is a "sweet spot" for depth, which could explain why architectures need things like skip-connections to manage this effect.

This RMT framework also provides a powerful lens through which to view other very recent theoretical approaches. For example, some work analyzes the NTK spectrum by studying it as a fixed point of composing kernels based on inverse cosine distances. That view explains *that* information is transformed layer by layer, while our RMT approach explains *how* this composition impacts the actual learnability of features by modulating the BBP threshold. It allows us to start asking much more precise questions: which layers contribute most to the final signal, and which just add noise that makes learning harder?

Okay so now I'm gonna show my theoretical and empirical results when investigating the NTK for finite width
I recall that the NTK regime where the optimization process is well described by the NTK requires
that we have a width that is polynomial in your dataset size,
but in practice we want deeper neural network, and to disentangle the ffect of depth with the 
effect of width

in fact, the NTK is not the only kernel that describe well the optimization dynamics, there are whole 
family of kernel that is indexed by the natural numebr, that we call the NTH 
and that allows you to compute theoretically and numerically what will be the true optimization path 
your training will have wrt your dataset and network

the way to construct them is by taking the dot product of the former kernel of the gradients wrt parameters of your network
with the former kernel, you can see the definition of the third order kernel now

for the NTK, wrt depth, we know theoretically (see the references) that the spectrum is linear in the depth
you can see some experiments i've done this week that confirm it. in fact the spectrum using the infinite width
ntk has 1 big eigenvalue and the others have the same magnitude as the minimum eigenvalue (bulk)
and scales linearly


for the other kernels, in fact, you can describe the NTK finite width correction with this late time correction
globally you can approximate your finite width ntk by adding this kernel that involes the O3 and O4 in the hierarchy
so from now we can do 2 things, we can just try to get scaling laws for the 1/M correction (where M in the depth)
or we can compute O3, by hand, numerically, and see what's happening

i've done both, the second is very much much more difficult and i'll explain why later,
now we will focus on just analyzing the NTK correction 1/M asymptotic expansion 


so experimentally I use Jax and the neural tangents library, I evaluate Kemp with neural tangents, and Kinf with
a formula i've presented 2 weeks ago with the cosine kernel, I use this setup, it was very very long to
run all of this, and I've many things to disentangle again in the computations but i've got some results

what I got is something like that, that the correction scales linearly (or super linearly) with N (dataset size) and L
and that remains bounded (which is a bit logical) wrt the input dimension

you can see a plot i've made, it took me 24h to run this, and you can see a superlinear growth with the upside
of the plot

you can see the same with D_in, and N


Now just to show you why I do that, because ith weyl inequality (that is very not tight) we can do something like that
to try to get a optimum bound for the smallest eigenvalue wrt depth, width and your parameter budget P
the calculations are not that interesting because i need to be confident with the linear or super linear growth
but in fact you can have something like that, you can be ensured to maintain a good spectrum if your width scale as your dataset size

overall the achievement now is great, because I can optimize a bit my experiments and get better results for the next week, i'm 
happy and confident with that, i've made a great code that runs well even a bit slow, but it is fully reproducible

but this is not very tight, and can ccontradict a bit what we do empirically when training deep neural nets, and that's why the best is to understand those kernels K3 and K4


by hand it's very hard to get a good formula, it tooks me several days to be sure of what i'm writing but you can
get a whole formula i've written it in the report for K3, and for K4 this formula take like 2 pages i've not written it now


from now i'm not a lot confident on what I'm saying but i'll try to make you understand the goal
we want to get scaling laws and trends for a big formula and we can try to infer it by analyzing its terms
that are backprop terms, derivatives, weigth matrices and forward prop terms
we can try to do some scaling analysis for those terms, and we can find something that scales between linearly and quadratically
for O3,, for O4 we can do the same, but for O4 there are a scaling of 1/lambda² with eigenvalues of the NTK, 1/lambda for O3
and we can guess that the O4 contribution has the same magnitude as O3


so this between linear or quadratic scaling which is also what I found for the correction term, so I think that 
what I tried to guess from analyzing my formula can give some insights, but it is not totally
rigorous because if we want so we need to compute lyapunov exponents for random gaussian matrices, we know it has
a log(M) trend but it's a huge work.


I've done abit of some experimental setups, that I'll run today, but from right now
it's computationnaly expansive but there are a lot of rewards because no one has ever done that


just to conclude, i'll compute the scaling of the O1 formula by hand with a mroe tight analysis for the correction
i'll investigate the O4 kernel, run extensive experiments for O3, optimize my implementation and publish it
because i've seen it nowhere
and i'll do what i've done with the NTK corrections in other setups (torus with what I said the last week) to get results
that can be applied with the DSRN framework

and also try to understand if we get the same when there are resnets or skip connections (because we have some better results)
from that point of view

I'll also try to do the same but for some very narrow network, to compare it with the deep narrow network theory, 

