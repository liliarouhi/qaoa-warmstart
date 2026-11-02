# Mathematical conventions

All code follows these definitions. If the thesis uses a different sign or order,
change it here first and note it in `decisions.md`.

## Max-Cut
Graph $G=(V,E,w)$, $|V|=n$, $|E|=m$, weights $w_{uv}\in\mathbb{R}$ (negative allowed, Lemma 1).

$$\mathrm{cut}(z)=\sum_{(u,v)\in E} w_{uv}\,[z_u\neq z_v],\qquad z\in\{0,1\}^n$$

$$C=\sum_{(u,v)\in E} w_{uv}\,\frac{1-Z_uZ_v}{2}\quad(\text{diagonal, } C|z\rangle=\mathrm{cut}(z)|z\rangle)$$

Bit ordering in code: basis index $k$ has $z_i=(k\gg i)\,\&\,1$.

## QAOA
$$|\boldsymbol\gamma,\boldsymbol\beta\rangle=\prod_{l=1}^{p} e^{-i\beta_l B}e^{-i\gamma_l C}\,|+\rangle^{\otimes n},\qquad B=\sum_j X_j$$
Layer $l=1$ acts first. Objective (maximised):
$$F_p(\boldsymbol\gamma,\boldsymbol\beta)=\langle\boldsymbol\gamma,\boldsymbol\beta|C|\boldsymbol\gamma,\boldsymbol\beta\rangle,\qquad r=F_p/C_{\max}$$
Parameter vector in code: $\theta=(\gamma_1,\dots,\gamma_p,\beta_1,\dots,\beta_p)$.

## Symmetries (Lemma 2) and fundamental domain
| Symmetry | Condition | Used for canonicalisation |
|---|---|---|
| $F(-\gamma,-\beta)=F(\gamma,\beta)$ | always | $\gamma_1\ge 0$ |
| $\beta_l\to\beta_l+\pi/2$ | always, each layer separately | $\beta_l\in[-\pi/4,\pi/4)$ |
| $\gamma_l\to\gamma_l+2\pi$ | integer weights only | $\gamma_l\in(-\pi,\pi]$ |

Proof hint for $\beta$: $e^{-i(\pi/2)X_j}=-iX_j$, so the shift multiplies by $(-i)^n\prod_j X_j$,
which commutes with $C$ and $B$ and fixes $|+\rangle^{\otimes n}$; hence it is a global phase.

## Weight normalisation
DVRP graphs: $w_{uv}=\|x_u-x_v\|/\max\|x_a-x_b\|\in(0,1]$. The factor is stored in
`graph.weight_scale` so the original instance can be recovered.
Open question: Sureshbabu et al. (2024) study how to rescale weights so that parameters
transfer between weighted instances. Read their rescaling rule and decide (decisions.md)
whether to adopt it instead of max-distance scaling before Phase 2. (Not yet checked in detail.)

## Cost metric
`nfev` = number of calls to $F_p$ (each is one full circuit evaluation). ETT = nfev until
$F_p\ge \tau\cdot F^{\text{best}}$. "Iterations" is not used (not comparable across optimizers).

## Complexity (state-vector simulation)
| Step | Time | Memory |
|---|---|---|
| cut diagonal (once per graph) | $O(m\,2^n)$ | $O(2^n)$ |
| one evaluation of $F_p$ | $O(p\,n\,2^n)$ | $O(2^n)$ |
| exact $C_{\max}$ | free after the diagonal ($O(2^n)$ max) | — |
| p=1 closed form, per evaluation | $O(m)$ after an $O(m\,d_{\max})$ table | $O(m)$ |
