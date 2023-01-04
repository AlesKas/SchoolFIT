(undirected graphical model
![[markov.png]](assets/markov.png)

## Clique
A subset of nodes where all nodes are connected directly to each other

## Maximal Clique
A clique where no more nodes can be added from the graph.

## Factorization
Joint probability distribution over random variables can be expressed as normalized product of potential functions $\psi_c(x_c)$, which are positive valued functions of subsets of variables $x_c$ corresponding to max cliques $C$.
Potential functions are expressed in form of energy functions $E(x_c)$, which means we sum up $E(x_c)$ and do not use dot product of $\psi_c(x_c)$.

$$P(x_1, x_2, x_3, x_4) = {1 \over Z} \psi_{1,2,3}(x_1, x_2, x_3) \psi_{2,3,4}(x_2, x_3, x_4)$$
$$= {1 \over Z} exp\{-E(x_1,x_2,x_3) - E(x_2,x_3,x_4)\}$$

## Conditional Independence
Works the same as in [[Conditional Independence]].
We marginalize out priors and and try to prove that given certain priors, the posteriors are independent.

## Inference on a chain
TODO