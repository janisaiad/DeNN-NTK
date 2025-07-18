Okay so this is some very fresh things I wanted to talk about , yesterday
I’ve written MMFN on latex and copy pasted it but it is MMNN along the whole presentation
Professor Shijun zhang wanted to get a theoretical understanding for the optimization point of view of MMNN 

to get to that we can investigate the global optimization process, that is for instance to get scaling law of the loss after training wrt depth and width

to recall, MMNN try to get the a low rank very shallow Matrix W and A, with W the random basis vectors and A is the trained coefficients

just to simplify the computations I've get rid off the resnet and residual connections to go to the most important thing that is the NTK for a 1 layer MMNN


the 1st thing to say is that the NTK is only valuable for training parameters, that is A and C, which means you need to introduce an NTK parametrization of A and C to make sure that the sums are converging, which means you scale the parameters of A not as the initialization with a uniform law, but a scaled uniform or gaussian law, with 1/sqrt(n) std, with n the model dimension, that is the number of lines of W growing to infty

that makes the dot product factorized by 1/n , and you can then make it converge to an expectation

i’ve skipped the computations because i did not written it in latex, it tools me 3 papersheets long but the most important thing is that inside the NTK, inside the activation function you keep the bias in the expectation, which is not the case for FCNN

if you have 2 entries, there are many things to transform the integrals from d dimensional with d the input dimension, toward 2 dimensionals, we integrate gaussian laws over a cone in the 2d space using dual variables, this is hard, i’ll make it very detailled for the next days

and at the end, you have a deviation of the NTK from the FCNN towards something that takes into account the correlations of 2 vectors (that is the angle between them) and the scaling of these vectors along some directions (that is in the a variable and sigma_2 variables),

so for the next days i’ll make it very clear, extend it to many layers, do a lot of numerical experiments and comparisons to have some numerical understanding of the optimization landscape of MMNN, but now the thing to say is that the mathematical content is rich, richer than FCNN.





FCMN : 
typo N^(L-l-1), also cauchy swarz at a moment
DSRN : 

Proof of the main theorem :
the approximation power is optimal, we use VCdims of laplacians because what’s delimititing the upper bound for vcdims is how we partition the parameters and how they partition the latent space with linear forms and quadratic forms
we care about the generalization error, according to bartlett we use rademacher (need to get a very good theoretical understanding)
generalization explode with M in general (M number of parameter, according to pr yang undersatnding)

can we lower bound rademacher complexity by covering number (dudley’s theorem inversed)

Pdim(omega) inferior to VCdim(omega-y)

essayer de formaliser complètement la preuvee DSRN



TODO : 
fix typo in overleaf
use sine/cos (laplace transforms for gaussians) and sintu, use 2 hidden layer and low rank inside
code in jax, neural tangents
see the spectrum difference between NTK, and compute for relu with arbitrary L
show the landscape and the NTK parameterization (and the hessian also)

maybe notebooklm
NTK for low rank normal mateices (by product ?) (its an algebraic variety using determinantal characterizaton through matrix minors)
(or the newton schultz/truncated svd of it because we know the spetcrum well , and use rmt stuff to sum through ntk/lyapunov)
je pense que les wavelets et FMNN snt très liés, parce que ce sont les memes facettes 
NTK for DSRN, same
