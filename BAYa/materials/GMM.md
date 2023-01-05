First some basics
$$p(x|\eta) = \sum_{c} \mathcal{N}(x; \mu{c}, \sigma_{c}^{2}) \text{ } \pi_{c}$$
$$\eta = \{\pi_c, \mu_c, \sigma_c^2\}$$
$$\sum_c \pi_c = 1$$
![[gmm.png]](assets/gmm.png)
## BN for GMM
The picture above combines multiple distributions. So, let's imagine each distribution is a categorical latent variable $z$. The generation of sample $x$ from above would then look like this.
$$p(x) = \sum_z p(x|z) \text{ } P(z) = \sum_c \mathcal{N}(x; \mu_c, \sigma_c^2) \text{ } Cat(z = c|\eta)$$

## Training
### Viterbi
Using current model parameters, assign data points to sub-distribution Gaussians of the GMM model. Re-estimate parameters of each sub-distribution based on these data points, weights correspond to data points that belong to each Gaussian. Repeat until solution converges.
### [[Expectation maximization algorithm]]