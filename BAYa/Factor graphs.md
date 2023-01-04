Pretty much same thing as [[Markov Random Field]], except factors (potential functions) are explicitly specified.

### Conversion
#### MRF to FG
Here's an example of a conversion from MRF to to a factor graph. As can be seen, FG is more explicate, so one MRF can represent more variants of FG
![[rmf-to-fg.png]](assets/rmf-to-fg.png)

#### BN to FG
[[Conditional Independence]] disappears during conversion - explain-away effect is gone. Just like with MRF, one MRF can represent more FGs, cause FG can display data that BN does not.
![[bn-to-fg.png]](assets/bn-to-fg.png)