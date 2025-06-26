"CONCISE"














"LONG"

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

