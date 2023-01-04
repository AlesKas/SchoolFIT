For $c=1,2 .. \infty$
1. Generate $v_c$ in range $(0,1)$ from $Beta(1, \alpha)$
2. Break the stick into two pieces with proportions $v_c : 1 - v_c$
3. Length of first part is $\pi_c$
4. Second part is broken again in further iterations
$$v_c = Beta(1, \alpha)$$
$$\pi_c = \mathcal{v}_c \prod_{k=1}^{c-1} (1-v_k)$$
![[gem-samples.png]](assets/gem-samples.png)