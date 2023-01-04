$$
ln\text{ }p(X|\eta) = \sum_{Z} q(Z) \text{ } p(X, Z|\eta) - \sum_{Z} q(Z) \text{ } ln \text{ } q(Z) - \sum_{Z} q(Z) \text{ } ln \text{ } {p(Z|X, \eta) \over q(Z)}
$$
$$
= \mathcal{Q}(q(Z), \eta) - H(q(Z)) - D_{KL}(q(Z)||p(Z|X,\eta)
$$

## Evidence lower bound ELBO
$$\mathcal{L}(q(Z), \eta) = \mathcal{Q}(q(Z), \eta) - H(q(Z))$$
## Non-negative entropy of distribution
$H(q(Z))$

## Auxiliary function
In EM there's a per observation auxiliary function being used:
$$\mathcal{Q}(q(Z),\eta) = \sum_n \mathcal{Q}_n(q(z_n), \eta) = \sum_n \sum_{z_n} q(z_n) ln \text{ } p(x_n, z_n|\eta)$$

## Kullback-Leibler divergence
$D_{KL}(q||s)$ measures unsimilarity between two distributions. $$D_{KL}(q||s) = 0 \Leftrightarrow q = p$$
## EM for continuous latent variable
Use integrals, otherwise same aproach

## q(Z)
Distribution over latent variable Z

## Steps
goal: find $\eta$ that maximizes $p(X|\eta)$

## E-step
Perform probabilistic data assignments of each data point to some class based on current hypothesis h.

$q(Z) := P(Z|X, \eta^{old})$
makes $D_{KL} = 0$
makes $\mathcal{L}(q(Z), \eta) = ln \text{ } p(X|\eta)$

## M-step
Update hypothesis h for distributional class parameters based on new data assignments

$\eta^{new} := arg_{\eta} \text{ } max \text{ } \mathcal{Q}(q(Z), \eta)$
$D_{KL}$ increases as $P(X|Z, \eta)$ deviates from $q(Z)$
$H(q(Z))$ does not change for fixed $q(Z)$ 
$\mathcal{L}(q(Z), \eta)$ increases like $\mathcal{Q}(q(Z), \eta)$
$ln \text{ } p(X|\eta)$ increases more than $\mathcal{Q}(q(Z), \eta)$

## Update of weights
Weights $\pi_c$ are required to sum up to 1. Lagrange multiplier $\mathcal{L}$ is used to enforce this rule while updating weights.
$$\pi_c = {\sum_n \gamma_{nc} \over \mathcal{L}} = {\sum_n \gamma_{nc} \over \sum_k \sum_n \gamma_{nk}}$$