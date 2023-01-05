statistical independence: $P(a,b) = P(a)P(b)$
conditional independence: $P(a|b,c) = P(a|c)$ - $a$ is independent of $b$ given $c$
or also $P(a,b|c) = P(a|b,c)P(b|c) = P(a|c)P(b|c)$

## Cheatsheet
1. We are searching for x, y. We have z fixed.
2. Calculate $P(x,y|z)={P(x,y,z) \over P(z)}$. On the upper half is the entire network architecture, on the lower half are the fixed variables.
3. If you get $P(x, y|z) = P(x|z)P(y|z$, then $x$ and $y$ are independent given $c$. Otherwise they are not.

> exercises calculated in notebook