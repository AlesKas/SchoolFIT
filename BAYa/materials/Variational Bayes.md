$$ln \text{ } p(X) = \int q(Y) \text{ } ln \text{ } p(X, Y) \text{ } dY - \int q(Y) \text{ } ln \text{ } q(Y) \text{ } dY - \int q(Y) \text{ } ln {p(Y|X) \over q(Y)} \text{ } dY$$
1. find $q(Y)$ which approximates the true posterior $p(Y|X)$
2. maximize $\mathcal{L}(q(Y))$ which in turn minimizes $D_{KL}(q(Y)||p(Y|X))$
- craft a distribution $q(Y|\eta)$ and optimize $\mathcal{L}(q(Y|\eta))$
The difference as opposed to expectation maximization is that in EM, $\eta$ output is a hard decision (one point estimate), while in case of VB it is a distribution. This means that measure of uncertainty is reflected in the output of a VB.

## Mean field approximation
Used to optimize VB

Split model variables $Y$ into subsets $Y_1$, $Y_2$, $Y_3$, .. with conditionally conjugate priors
$p(Y_i|X, Y_{\forall j \neq i})$
