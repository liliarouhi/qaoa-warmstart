# Conjecture E: relative angle shift O(1/n) at p = 1 (dense DVRP graphs)

Status (2026-11-02): **partly proven, main claim open.** Lemma E1 is proven (and checked to 1e-15).
Proposition E2 is a formal leading-order calculation (not a rigorous asymptotic proof). Lemma E3 is proven
given E2's leading-order form. The final claim (Conjecture E) is supported numerically (E14) but NOT proven.
AI assistance: derivations and numerics were prepared with Claude (Anthropic); the author must check and rewrite them.

Setting: complete graph on n points in the unit square, weights w_uv = |x_u - x_v| / sqrt(2) in [0, 1]
(fixed scale, D8), cost C = sum_{u<v} w_uv (1 - Z_u Z_v)/2, p = 1, F(gamma, beta) = <C>.
Event: one new uniformly random point is added. theta* = (gamma*, beta*) is the optimum before,
theta'* the optimum after (reached from theta*, local tracking as in E11).

## Lemma E1 (weighted p = 1 closed form)
For every edge (u, v), with k running over all other vertices,

    <C_uv> = w/2 + (w/4) sin(4 beta) sin(gamma w) [ prod_k cos(gamma w_uk) + prod_k cos(gamma w_vk) ]
                 - (w/4) sin(2 beta)^2 [ prod_k cos(gamma (w_uk - w_vk)) - prod_k cos(gamma (w_uk + w_vk)) ]

where w = w_uv. For unit weights it reduces to Wang et al. (2018) Lemma 3 (a common neighbour gives
cos 0 = 1 and cos 2 gamma; a neighbour of only one end gives cos gamma in both products).
*Proof idea:* the same Heisenberg-picture computation as Wang et al.: Z_u Z_v conjugated by the mixer gives
terms in Z_u Z_v, Y_u Z_v, Z_u Y_v, Y_u Y_v; conjugating by e^{-i gamma C} only the factors
e^{-i gamma w_uk Z_u Z_k} touching u or v act, and each contributes one cosine factor (or the
cos(a +- b) pair for Y_u Y_v). The author should write this out; it is a direct generalisation.
*Check:* equals the state-vector simulator to 1e-15 (n = 7, three angle pairs) and `tests/test_lemmas.py`
(to be added: `analytic_p1.expectation_p1_weighted`).
*Literature:* a weighted version very likely exists (Ising p = 1 formulas). Search before claiming novelty;
no reference is given here because none was verified.

## Proposition E2 (leading order: only the product gamma * beta matters)  — formal
Put s = n gamma beta and let gamma, beta -> 0 with gamma sqrt(n) -> 0. Expanding Lemma E1
(sin x ~ x, prod cos ~ 1, and prod cos(gamma(a-b)) - prod cos(gamma(a+b)) ~ 2 gamma^2 sum_k a_k b_k):

    F - W/2  ~  (2 s / n) S_2  -  (2 s^2 / n) T,      S_2 = sum_{u<v} w_uv^2,
                                                      T   = sum_{u<v} w_uv c_uv,  c_uv = (1/n) sum_k w_uk w_vk

So to leading order F depends on (gamma, beta) only through the product s: the landscape has a
**flat valley** gamma * beta = s*/n with

    s* = S_2 / (2 T).

For uniform points in the unit square: s* -> 1.62 (n = 1500: 1.616). Observed at finite n (E14):
n gamma* beta* = 1.71 (n = 16) ... 1.82 (n = 96); the 10% gap comes from the neglected terms
(gamma sqrt(n) is about 1.5 at these n, not small).
*Consequence:* the Hessian at theta* has one stiff direction (across the valley) and one weak direction
(along it, almost exactly the log-gamma direction: |v_weak . e_log gamma| = 0.997). Numerically, in coordinates
(log gamma, beta): lambda_weak ~ n^0.65, lambda_stiff ~ n^1.74. This is why the isotropic bound of
Theorem C (shift <= eps / mu, mu = smallest curvature) does NOT decay with n (eps/mu ~ 0.3 for n = 16..128):
it ignores that the perturbation gradient is small along the weak direction.

## Lemma E3 (the valley moves only O(1/n))
Assume weights in [0, 1], mean weight >= w_0 > 0, and T >= c' n^2 (see proof). Adding one vertex changes S_2 and T by at most
n and 3n/2 + O(1) respectively, while S_2 >= c n^2 and T >= c' n^2. Hence

    | s*_{n+1} - s*_n | / s*_n = O(1/n).

*Proof:* the new vertex adds n edges (each w^2 <= 1) to S_2; in T it adds n edge terms (each <= 1) and changes
every c_uv by (w_ux w_vx)/(n+1) - c_uv/(n(n+1)) = O(1/n), over n^2/2 edges with weight <= 1, i.e. O(n) in total.
Lower bounds: S_2 >= (sum w)^2 / (n^2/2) >= w_0^2 n^2 / 2 (Cauchy-Schwarz). For T we ASSUME T >= c' n^2
(true when the points are spread out, e.g. uniform in the square; it fails only if almost all weights are ~0). The ratio of two
quantities each changing by relative O(1/n) changes by relative O(1/n).  QED (given E2).

## Conjecture E (open)
At p = 1, for dense Euclidean DVRP graphs,  |gamma'* / gamma* - 1| = O(1/n)  and  |beta'* - beta*| = O(1/n).
What is proven covers the stiff direction (the product, Lemma E3). The position ALONG the valley is fixed by
higher-order terms (gamma sqrt(n) of order 1, the damping factors prod cos), and a proof would need a
uniform expansion of those terms plus a curvature bound for the 1-D valley problem. Not attempted further.

## Numerical evidence (E14)
40 random graphs per n, exact optimum from the closed form (grid + Nelder-Mead polish), one new point:
median relative gamma shift 0.033 (n = 16), 0.021 (24), 0.017 (32), 0.012 (48), 0.0078 (64), 0.0056 (96);
log-log slope **-0.99** (n = 16..96). beta shift slope -1.21. Earlier E11 (simulator, n = 6..18): -1.07.
Correction to E11's heuristic: gamma* does NOT shrink like 1/n: n gamma* grows (7.9 at n = 16, 14.8 at n = 96),
roughly gamma* ~ n^-0.64 and beta* ~ n^-0.33 in this range, with the product gamma* beta* ~ 1.8 / n (E2).
