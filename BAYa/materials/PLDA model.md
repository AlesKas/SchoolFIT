Used for speaker verification, speakers are represented by high dimensional embedding vectors.
We are assuming same factorirazation as for GMM, but with continuous variable z.
$$p(z) = \mathcal{N}(z|\eta, \Sigma_{ac})$$
$$p(x|z) = \mathcal{N}(x|z, \Sigma_{wc})$$
Embeddings are assumed to be generated as follows:

1. Latent (speaker mean) vector $z_{s}$ is generated for for each speaker $s$ from gaussian distribution $p(z)$
2. All embeddings of speaker $s$ are generated from gaussian distribution $p(x_{si}|z_{s})$

![PLDA](assets/plda.png)
## Same/different speaker hypothesis
![plda-hypothesis](assets/plda-hypothesis.png)
