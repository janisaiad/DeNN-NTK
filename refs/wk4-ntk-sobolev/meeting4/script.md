To say : 
today i will talk about what i’ve learned by reading some papers,
what I found and the links i’ve made through this reading, i’ve done some computations by hand, this weekend i will do some experiments and i’m confident on what i’ve understood

so 1st the initial question was is deeper or wider better for sobolev training under an NTK’s perspective

For wide networks, the NTK is fairly well known, especially in the optimal setup where the initialization keeps the variance constant through layers (which is know as EOC)

we have a closed form of the NTK which is this formula and for a particular dataset like 3 points, you have an NTK matrix.

And what dictates the training is the spectrum of this NTK matrix.

You can see that the NTK depends linearly on the L2 norms of your data, and that’s why in general you normalize your dataset to be on the d dimensionnal sphere
In fact, if you do not this, it does not matter a lot for the spectrum because we want a scaling law of the spectrum with respect to L up to a multiplicative constant (L2 norm of x) because of the homogeneity of ReLu

The other thing is that the NTK, when you have enough data (like a distribution), gives you an operator, a PSD operator, eigenfunctions are spherical harmonics where the eigenvalues decay polynomially

When you are in the setup of sobolev training, the loss is modified and what you are really doing is a gradient descent wrt the symmetric operator that multiply in the fourier space.
So it’s a least square wrt to another norm


In the sobolev training setup, this NTK matrix is multiplied by P that comes from going back in the fourier space with the sobolev loss



To answer the question we have to understand before how the spectrum of the operators behave (before discretizing it with our samples).


Sobolev training is something truly general, in fact you can take any exponent s, you get what is called a fractional laplacian (from the laplace beltrami operator on the sphere, it is very general because it comes from differential geometry)). the eigenvalues remains the spherical harmonics and the spectrum scales as k^2s to compensate the spectral decay of the NTK operator

What’s happening is that you can learn differently the frequencies of your function and if s is bigger, you learn high frequencies before.

That’s the theory behind sobolev training on the sphere  that can be extended to any distribution over the sphere. the main thing is that you need to diagonalize first the NTK by computing the quadrature weights numerically , and you get eigenfunctions (but different eigenvalues). The polynomial decay remains present and you still can counterbalance it when you train a neural network.

So that’s for the operator, and this is in the limiting setup of many normalized data.

To answer the question deeper or wider right there, there are some results from francis bach (i’ve made a pdf report also with the references therein) where there is a multiplicative constant on the spectrum that depends linearly with L. And that’s promising because it is not the only time we see a linear dependency of the spectrum wrt L


In fact the other thing we want to understand is the NTK matrix that comes from you samples.

before all we recall the initialization for a leaky relu function and the covariance map between layers (with rho the cosine distance between x and x prime) gives you that

so I recall the Deep NTK decay for the operator spectrum has a linear dependency wrt to L
But when you have enough data, and there is something to disentangle here, the spectrum of the matrix approximate the spectrum of the operator, but we want to calculate before a limit with l growing to the infinity with a number of data that is constant or big


in this setup the major paper that is very very recent (MLP at the EOCs) i’ve worked on this paper everyday tells you that what matters the most when you have a wide network and you make L growing, the NTK matrix is quasi-affine wrt the inverse cosine distance matrix through the layers - and so the spectral bounds

the results are that the NTK matrix spectrum scales as 3*l/4 for the 1st eigenvalue,
and 3*l/(4n) for the last eigenvalue. and that’s promising a lot like this results are just 3 months old to get the deep understanding, but there were another proof of this result that is very old and very hidden in the literature (that comes from the papers “disentangle trainability and generalization”).

So know we understand well what to do to get a big enough smallest eigenvalue of the NTK matrix by getting enough data.
And what to do next is to understand how this inverse cosine distance matrix behaves when being multiplicated by P (both are rotation invariants) so the most promising path for me for the next days is to perform this calculation and see what’s going on, if the spectral bounds are getting better with the number of layers. that’s new, 



Here the setup is a wide then deep network or wide then getting a scaling law wrt L (no very very deep networks)

But we know that the NTK also exists for deep narrow networks from the eponymous paper


I’ve done the computations by hand, and you see that the NTK of this deep narrow network is very simple like it behaves a bit like a 2 layers network, so with this initialization that is very special from this paper (like there is not a hugh stochasticity and a lot of zeros)
It is not rotation invariant because of the bias but i’ve made a limited development for bias small against weights (beta a lot smaller than rho) and we see a behaviour that is not so promising. because a huge part of the weights do not contribute to the NTK at the initialization because their init comes from an universal approximation theorem for networks that are a+b+1 wide (with a the input dimension and b the output dimension)


from that we can do 2 things, like concatenate those deep narrow networks, that will compose the NTK as the big formula for the operator we got before, we can do a mean field analysis that i won’t detail but it has already been done for big resnets with other initialization , we can test some different initialization schemes and not to put many zeros in the initialization,  see other architectural modfis like skip connections etc .. there is many things to do.



Another promising path for the next days is that, the sobolev operator you get inherits from a kernel structure because of summations identities for spherical harmonics if the weights are uniform (and you can find also other formulas for other weights by computing orthogonalizing the with respect to another distribution from a spherical scalar product)
i’ve seen this today, i think there is something to do interesting because it could simplify a lot the spectral analysis of KP the product of NTK matrix and P matrix
















