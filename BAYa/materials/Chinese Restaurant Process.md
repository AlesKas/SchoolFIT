The problem:
	Prior is infinite weight vector from $GEM(\pi|\alpha)$
	$z_n$, $n=1..N$ are samples generated from unknown $Cat(z_n|\pi)$
	Posterior is intractable because prior is infinite

However, predictive posterior can be evaluated as
$$P(z'=c|z) = {N_c \over \alpha+N}$$
$$P(z'=C+1|z) = {\alpha \over \alpha+N}$$
$N_c$ is the number of observations assigned by $z$ to category $c$.
$C+1$ is the new, yet unseen category.

It's an example of [[Predictive Posterior]]?