# Theorem C: how far the optimal angles move when the graph changes

> **Authorship note.** Drafted with AI assistance (Claude) on 2026-11-02 at the author's request.
> The author must check every step, rewrite it in her own words, and disclose AI assistance as IUST
> rules require. Numerical check: `tests/test_angle_shift.py`. Thesis placement: Chapter 3.
> The argument is a standard strong-concavity perturbation bound; the contribution is applying it,
> with Lemmas A/B, to dynamic QAOA and measuring the constants (Chapter 6).

## Setting
$F=F_G$ and $F'=F_{G'}$ are the objectives before and after the change (old graph embedded with the new
vertex isolated, Lemma A0, so both are functions of the same $\theta\in\mathbb{R}^{2p}$). Both are smooth
(trigonometric polynomials in $\theta$). $\theta^*$ is a local maximiser of $F$.

## Theorem C
Assume that on the ball $\mathcal B=\{\theta:\|\theta-\theta^*\|\le r\}$:
1. (curvature) $\nabla^2F'(\theta)\preceq-\mu I$ for some $\mu>0$;
2. (small gradient change) $\varepsilon:=\|\nabla F'(\theta^*)\|=\|\nabla(F'-F)(\theta^*)\|$ satisfies $2\varepsilon/\mu<r$.

Then $F'$ has exactly one local maximiser $\theta'^*$ in $\mathcal B$, and
$$\|\theta'^*-\theta^*\|\le\frac{\varepsilon}{\mu}.$$

*Proof.* (The equality in 2 uses $\nabla F(\theta^*)=0$.)

*Existence.* Let $\rho=2\varepsilon/\mu<r$. For $\theta$ on the sphere $\|\theta-\theta^*\|=\rho$, Taylor's theorem with
assumption 1 gives
$$F'(\theta)\le F'(\theta^*)+\langle\nabla F'(\theta^*),\theta-\theta^*\rangle-\tfrac{\mu}{2}\rho^2\le F'(\theta^*)+\varepsilon\rho-\tfrac{\mu}{2}\rho^2=F'(\theta^*).$$
(With $\rho$ slightly larger than $2\varepsilon/\mu$ the inequality is strict; take any such $\rho<r$.) So the maximum
of $F'$ on the closed ball of radius $\rho$ is attained in the interior, at a critical point $\theta'^*$.

*Uniqueness.* $F'$ is strictly concave on $\mathcal B$, so it has at most one critical point there.

*Distance.* Strong concavity gives, for $a,b\in\mathcal B$,
$\langle\nabla F'(a)-\nabla F'(b),a-b\rangle\le-\mu\|a-b\|^2$. Put $a=\theta^*$, $b=\theta'^*$ ($\nabla F'(b)=0$):
$\mu\|\theta^*-\theta'^*\|^2\le\langle\nabla F'(\theta^*),\theta'^*-\theta^*\rangle\le\varepsilon\|\theta'^*-\theta^*\|$. Divide. $\square$

## Bounding $\varepsilon$ (connects Theorem C to Lemmas A and B)
$\varepsilon$ is the gradient of $F'-F$ at $\theta^*$. The same steps as in Lemma A, applied to
$\partial_{\theta_k}F=2\,\mathrm{Im}\langle\lambda|G_k|\phi\rangle$ (adjoint formula, `simulator.gradient`), give a bound
of the form $\varepsilon\le c_1\|\Delta C\|\,(1+c_2\|C\|\textstyle\sum_l|\gamma_l|)$ with constants depending on $p$ and $n$
(**sketch; full constants not yet derived**). In the sparse regime only the terms in $A$ (Corollary B2) contribute,
so $\varepsilon=O(D^{p+1})$. In practice $\varepsilon$ and $\mu$ are measured directly (below), which is what the thesis
needs to compare the regimes.

## Numerical check (E3, 2026-11-02)
40 complete Euclidean graphs, $n=6\to7$, $p=1$, fixed weight scale. Assumption 1 held on the segment in 38 cases.
Observed shift $\|\theta'^*-\theta^*\|$: median 0.056 rad. Bound $\varepsilon/\mu$: median 0.155. Ratio shift/bound:
median 0.31, max 0.68. So the bound held every time and is about 3x conservative; it is much tighter than Lemma A
(~14x) because it uses the local curvature.

## Remarks for the thesis
1. Assumption 1 can fail: at a degenerate or flat optimum ($\mu\approx0$) the optimum can jump. Report how often.
2. Practical meaning: if $\varepsilon/\mu$ is small, the previous optimum is a good start (temporal warm start); this is
   the theoretical reason behind RQ3.
3. The cache (C2) uses the same idea with "nearest stored graph" instead of "previous snapshot".
