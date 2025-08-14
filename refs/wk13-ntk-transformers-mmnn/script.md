




















Hi everyone, today i'll present what I found during the last days in a very structured presentation.
I've worked on 2 things, the 1st thing i've worked on was the numerical investigations to understand NTK of MMNNs
and how its randomness, randomness that appears due to the low ranks, how it scales wrt the low rank
I'll then present the theoretical material under the hodd, that meands how to derive a fully complete mathematical expression of the
NTK for MMNNs.
I'm focusing on 2 layers MMNs, and try to compare, on the theoretical point (by comparing the formula involved),
with it's fully connected counterpart. I'll talk a bit about the approximation power of kernels involved
and the low frequency bias, that pr zhang already know in its paper "Why Shallow Networks Struggle to Approximate
 and Learn High Frequencies"

I'll then finish by saying a few word about what i've also done about transformers, how to tackle this case with references
and i'll present another how i've derived the full expression for 1 layer attention and transformer

For the 1st part I recall you the setup for MMNNs



Now for the 2nd part
I'll show you all the hand calculations i went through, it was very hard and very long, b

I'm glad to present to you my hand calculations, because since the beginning of my work over the NTK, i've written about 300 pages of hand calculations. This is a retranscription of what i've done only during 3 days, with also reading papers, implementing numerical stuff and 
do comparisons etc ..



## tofill

Alright, for the theoretical part of my work, I focused on deriving a complete mathematical expression for the Neural Tangent Kernel of a two-layer MMNN. This was a long process, and I want to walk you through the key steps and ideas, following the path of the actual derivation.

**Part 1: The Foundation - The One-Layer Kernel**

Everything starts with the basic building block: calculating the NTK for a single layer. Basically, this means we need to compute a specific expectation: the expected value of the product of two ReLU functions, applied to a pair of correlated Gaussian variables.

Think of it like this: we have two inputs, `x` and `x'`, and the first layer's outputs before activation are Gaussian processes. We need to understand how their activations `ReLU(X_1)` and `ReLU(X_2)` are correlated. This expectation is the most important part of the one-layer NNGP kernel.

To solve this, we cannot do it in one single step. The standard method is to break it down. We separate the main integral into a sum of four simpler pieces, which I've called `I_11`, `I_12`, `I_21`, and `I_22`.
*   `I_11` is the easiest – it's simply the probability that both Gaussian variables are positive, which we can get from the bivariate normal CDF.
*   `I_12` and `I_21` are the first-order moments. They require more work, using some smart changes of variables and properties of the Gaussian density.
*   And then there's `I_22`, the cross-moment. This one is the real challenge. It's the most complex part of the calculation.

**Part 2: A Difficult Path - A Lesson in Complexity**

I want to briefly talk about an approach for `I_22` that *didn't* work. I think it's interesting. The idea was to use a nice identity with the partial derivatives of the Gaussian PDF. It looks good on paper. You put it into the integral for `I_22`, hoping it will simplify things.

But the opposite happens. You create a new integral that is even more complex. You try to simplify that, and the problem just gets more and more difficult. It was a dead end, but a good lesson: the most direct path isn't always the right one.

**Part 3: A Simpler Case - The Zero-Mean & Price's Theorem**

So, let's start again. A common strategy in math and physics is to first solve a simpler version of the problem to understand it better. What if we assume our Gaussian variables have zero mean?

Suddenly, the picture becomes very clear. The whole expectation simplifies a lot, and we get a beautiful, well-known formula: the **arccosine kernel**.

The best way to get this result is by using a powerful tool called **Price's Theorem**. In simple terms, Price's Theorem gives you a way to find the derivative of an expectation with respect to the correlation `rho`. For our ReLU product, this derivative is something very simple: the probability `I_11` we saw earlier!

So, the strategy is:
1.  Differentiate our expectation with respect to `rho` using Price's Theorem to get a simple expression.
2.  Integrate this expression back from 0 to `rho`.
3.  Add the integration constant, which is just the value of the expectation when `rho` is 0 – the case where the variables are independent.

This process gives us the arccosine kernel formula, and it's also the key to solve the general case.

**Part 4: The Main Part - The Two-Layer MMNN Kernel**

Okay, now for the main part. We have understood one layer. Let's go deeper and add another one. First, let's be very clear about the architecture we're analyzing. It's a network with two hidden layers:
1.  A first wide layer, just like in a standard neural network.
2.  A narrow "bottleneck" layer with a small, fixed rank `r`.
3.  A second wide layer.

The key idea is that the width `N` of the wide layers goes to infinity, but the rank `r` of the bottleneck stays fixed and small. We focus on 2-layer MMNNs to compare them with standard 2-layer Fully-Connected Networks (FCNNs). The big difference is that FCNNs have a huge weight matrix in the middle, with a number of weights that grows quadratically with the width. In our MMNN, because of the low-rank bottleneck, the number of weights only grows linearly.

Now, how do we compute the NTK for this? The formula for deep networks is recursive. The kernel for a layer `l`, which we call `Theta^(l)`, is built using the kernel from the previous layer, `Theta^(l-1)`. The general formula looks like this:

`Theta^(l) = Theta^(l-1) * (a term with derivatives) + (the NNGP kernel for layer l) + 1`

Let's apply this step-by-step.
*   **For the first layer (`l=1`)**: We start with `Theta^(0) = 0`. So the formula simplifies to `Theta^(1) = Sigma^(1) + 1`, where `Sigma^(1)` is the standard NNGP kernel for one layer. Since the inputs `x` and `x'` are fixed, this `Theta^(1)` is a deterministic kernel. It's the arccosine kernel we talked about. No randomness here.

*   **For the second layer (`l=2`)**: This is where things get really interesting. The inputs to the second hidden layer are not the original `x` and `x'` anymore. The inputs are now `h^(1)(x)` and `h^(1)(x')`, which are the *outputs* from the first bottleneck layer.

And because this bottleneck layer has a finite, fixed rank `r`, these outputs `h^(1)` are **random vectors**. Their values depend on the specific random initialization of the weights in the first layer.

This has a huge consequence. When we write down the formula for `Theta^(2)`, it now depends on these random vectors `h^(1)`.

This means `Theta^(2)` is no longer a fixed, predictable kernel that just depends on the inputs. It is a **random field**. Its value changes depending on the specific random initialization of the network weights. This is the key feature that makes MMNNs different from standard, infinite-width networks.

To make the calculations possible, we assume the biases are zero. We also need to be careful about how the signal propagates. We don't want the vector norms to explode or vanish as they pass through the layers. This is called staying at the "Edge of Chaos". To ensure this, we need to set the variance of the weights in a specific way, which introduces a normalization factor of `1/r`.

After all the calculations, we arrive at the final expression for the 2-layer kernel. It's a beautiful formula that depends on two things:
1.  `rho`: the correlation of the original inputs, `x` and `x'`.
2.  `rho_1`: the **random** correlation between the outputs of the first layer, `h^(1)(x)` and `h^(1)(x')`.

The formula clearly shows how the final kernel is a mix of different parts: one part comes from the first layer's kernel, and the other part is the NNGP kernel of the second layer, but this second part is random. It's random because it depends on `rho_1` and also on the random norms of the first layer's outputs, `||h^(1)(x)||` and `||h^(1)(x')||`.

So, we have an exact formula, but it contains random variables. The next logical question is: can we understand these random variables? And the answer, as we'll see, is yes.

This is the key takeaway: the NTK of a multi-layer MMNN is a random object, and we have an exact mathematical expression that captures this randomness through the correlation and norms of the intermediate layer's outputs.


At the end, you can see that if we do a good scaling, we have this multiplication, by something between 0 and 1, and that is promising
for getting NTK scaling law wrt depth, because we multiply some random variables and can have a geometric convergence

You can see if I remove r for this part and do a good scaling also,, when adding a depth we have a geometric divergence
with the products of ranks, as i've already shown in 1 month old experiments. This is very very promising i'm very happy of that.

I can also state that the numerical values are near the same in my experiments for high ranks. near 1 with a standard deviation
that decay fast, computing the scaling exponent exactly can be hard because the kernels have intractable forms for the 
std with the kibble and fisher distribution.

**Part 5: Understanding the Randomness - Fisher and Kibble Distributions**

So, our final expression for the two-layer kernel has these new random variables in it:
1.  First, there's `rho_1`, which is the random correlation between the outputs of the first layer.
2.  Second, there's the product of the norms of these outputs, `||x_1|| * ||y_1||`.

It looks complicated. How can we analyze a kernel that depends on random variables? Here comes the nice part. It turns out, we know the *exact* probability distributions for these variables.

The squared norms, `||x_1||^2` and `||y_1||^2`, together follow a **Kibble distribution**. And the random correlation, `rho_1`, follows a **Fisher distribution**.

And the most amazing part? The orientation of the vectors, which is related to the correlation `rho_1`, is completely **independent** of their lengths, which are the norms. This statistical independence is a wonderful result that makes the whole analysis possible.

**Part 6: The Conclusion - No Curse of Dimensionality**

So what does this all mean, especially when we think about the rank `r` of that bottleneck layer?

I studied what happens to these Fisher and Kibble distributions when the rank `r` gets large. And the result is fantastic.
*   The Fisher distribution for the correlation `rho_1` becomes extremely focused around its mean value. The randomness quickly disappears.
*   The Kibble distribution for the norms also becomes focused. If we look at the normalized product of the norms, `||x_1||*||y_1|| / r`, it converges to 1.

The conclusion is this: there is no "curse of dimensionality" here. As the rank `r` of our bottleneck layer grows, the random parts of our NTK become more predictable and concentrate around their mean values. The randomness is controlled, and the signal travels through the network in a stable way.

So, to finish this part: we now have a complete, analytical formula for the two-layer MMNN NTK. It's a random kernel, but its randomness is perfectly described by these beautiful distributions, and it behaves very well in high dimensions. This gives us a strong theoretical reason to understand why these architectures are so powerful.


Ok so now i'm talking about transformers
It has been very hard but I can state that I've derived the NTK for attention parts (under the 1/sqrt(d) scaling and infinite heads) in transformers, layernorm & res connections & FC layers are easy to manipulate then.
It's a rotation invariant kernel (zonal), as in FCNN. And it's great because we can apply a lot of stuff already use to derive the spectrum scaling (for the operator see 2009.14397 page 5 thm 1 eq 10/11)

In fact the goal is to derive the smallest eigenvalue & condition number, by applying what is stated in The spectrum of kernel random matrices (equation on K page 10).
You can see in this paper that this matrix structure aI + b11.T was also exactly what I found for MMNNs in my numerical experiments that i've presented the last weeks. I'm very happy because every part of the puzzle
seems


at the end there is an intractable integral like an integral and the denominator is a sum of exponential 'appear in soft max)
makes it impossible to integrate, so we get a special function;
to compute the spectrum of the ntk matrix with this special function in it
to have a great results