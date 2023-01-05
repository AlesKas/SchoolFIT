Let's say we can't sample directly $p(z_1, z_2)$, we can sample only from $p(z_1|z_2)$ and $p(z_2|z_1)$.

1. Initialize $z_1^*$ to any value
2. Generate $z_2^*$ using $z_2^* \sim p(z_2|z_1)$
3. Generate $z_1^*$ using $z_1^* \sim p(z_1|z_2)$
4. Go back to step 2

After a few iterations (burn-in phase), we will start getting $z_1^*$ and $z_2^*$ that are valid samples from $p(z_1, z_2)$.
Works for discrete/continuous and also scalars/vectors.

## Collapsed Gibs Sampling
Iterates over and samples from a subset of latent variables, eg discrete ones.
Integrates (marginalizes) over the rest of them (the continuous ones).