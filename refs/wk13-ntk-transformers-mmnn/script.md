

in terms of approximation, the low frequency bias 


















Make a more structured presentation : 








It has been very hard but I can state that I've derived the NTK for attention parts (under the 1/sqrt(d) scaling and infinite heads) in transformers, layernorm & res connections & FC layers are easy to manipulate then.
It's a rotation invariant kernel (zonal), as in FCNN. And it's great because we can apply a lot of stuff already use to derive the spectrum scaling (for the operator see 2009.14397 page 5 thm 1 eq 10/11)

In fact the goal is to derive the smallest eigenvalue & condition number, by applying what is stated in The spectrum of kernel random matrices (equation on K page 10).
You can see in this paper that this matrix structure aI + b11.T was also exactly what I found for MMNNs in my numerical experiments that i've presented the last weeks. I'm very happy because every part of the puzzle
seems


at the end there is an intractable integral like an integral and the denominator is a sum of exponential 'appear in soft max)
makes it impossible to integrate, so we get a special function;
to compute the spectrum of the ntk matrix with this special function in it
to have a great results