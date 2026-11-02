# Proposition D: the first-order (Newton) correction of the angles, and Lemma D0

> **Authorship note.** Drafted with AI assistance (Claude) on 2026-11-02 at the author's request.
> The author must check every step, rewrite it in her own words, and disclose AI assistance as IUST
> rules require. Numerical checks: `tests/test_perturb.py`, experiment E4 (`docs/experiments.md`).
> Code: `src/qws/perturb.py` (`newton_warm_start`, `refine_optimum`). Thesis placement: Chapter 3.
> Proposition D is the standard local error estimate of one Newton step; the contribution is using it
> as a warm start for dynamic QAOA and as the base of residual learning (C1). Lemma D0 is, as far as
> we know, our own observation (not found in the references checked so far; **verify before claiming novelty**).

## Setting
As in Theorem C: $F$ (old graph, new vertex isolated) and $F'$ (new graph) are smooth functions of
$\theta=(\gamma,\beta)\in\mathbb R^{2p}$, $\theta^*$ is a local maximiser of $F$, $\theta'^*$ the nearby maximiser of $F'$.
Write $g=\nabla F'(\theta^*)$, $H=\nabla^2F'(\theta^*)$, $\delta=\theta^*-\theta'^*$ (the "shift").

The **Newton warm start** is
$$\hat\theta=\theta^*-H^{-1}g .$$
It is the first-order (implicit-function) correction: it solves the linearised optimality condition
$\nabla F'(\theta^*)+H(\theta-\theta^*)=0$.

## Proposition D
Assume on the ball $\mathcal B$ of Theorem C (containing $\theta^*$ and $\theta'^*$):
1. $\nabla^2F'\preceq-\mu I$ (as in Theorem C);
2. the Hessian is $M$-Lipschitz: $\|\nabla^2F'(a)-\nabla^2F'(b)\|\le M\|a-b\|$.

Then
$$\|\hat\theta-\theta'^*\|\;\le\;\frac{M}{2\mu}\,\|\theta^*-\theta'^*\|^2\;\le\;\frac{M\,\varepsilon^2}{2\mu^3},\qquad \varepsilon=\|\nabla F'(\theta^*)\|.$$
So reusing the old angles has error $\|\delta\|$, while one Newton step has error $O(\|\delta\|^2)$.

*Proof.* Taylor's theorem with integral remainder for $\nabla F'$ around $\theta^*$:
$$0=\nabla F'(\theta'^*)=g-H\delta+R,\qquad R=\int_0^1\big(\nabla^2F'(\theta^*-s\delta)-H\big)(-\delta)\,ds .$$
By assumption 2, $\|R\|\le\int_0^1 Ms\|\delta\|^2ds=\tfrac M2\|\delta\|^2$. Hence $g=H\delta-R$ and
$$\hat\theta-\theta'^*=\delta-H^{-1}g=\delta-H^{-1}(H\delta-R)=H^{-1}R .$$
By assumption 1 every eigenvalue of $H$ is $\le-\mu$, so $\|H^{-1}\|\le1/\mu$ and $\|\hat\theta-\theta'^*\|\le\frac{M}{2\mu}\|\delta\|^2$.
The second inequality is Theorem C ($\|\delta\|\le\varepsilon/\mu$). $\square$

### A computable (crude) value of $M$
Centre the cost: $\tilde C=C-\frac{C_{\max}+C_{\min}}2I$, so $\|\tilde C\|=\Delta:=\frac{C_{\max}-C_{\min}}2$. Replacing $C$ by $\tilde C$
changes $F$ by a constant and the circuit by a global phase, so no derivative changes. Each parameter
$\theta_k$ appears once, as $e^{-i\theta_kG_k}$ with $G_k\in\{\tilde C,B\}$, $\|B\|=n$. In the Heisenberg picture, the
derivative in $\theta_k$ inserts one commutator $i[G_k',\cdot\,]$ with a unitarily conjugated $G_k'$ (same norm), so
$$|\partial_i\partial_j\partial_kF|\le 8\,\Delta\,\|G_i\|\|G_j\|\|G_k\| ,$$
and, bounding the operator norm of the third-derivative tensor by its Frobenius norm,
$$M\le 8\Delta\Big(\sum_k\|G_k\|^2\Big)^{3/2}=8\Delta\big(p\Delta^2+pn^2\big)^{3/2}.$$
**This is very loose** (E4: the observed constant is ~1000x smaller). It shows that $M$ is finite and polynomial in
$n,p$ and the weight scale; the thesis should use the measured constant.

## Lemma D0 (a new vertex moves the angles only at second order)
Let the new vertex $v$ have edges with weights $t\,w_{vj}$ ($t$ = a scale, $t=1$ the real order), and let
$F_t(\theta)$ be the QAOA objective of that graph ($F_0=F$: $v$ isolated). Then for all $\theta$
$$F_t(\theta)=t\,\tfrac{W_v}{2}+E_t(\theta),\qquad E_{-t}=E_t,\qquad W_v=\textstyle\sum_jw_{vj}.$$
Consequently $\nabla_\theta F_t=\nabla_\theta F+O(t^2)$, $\varepsilon=O(t^2)$, and (Theorem C) the shift is
$\|\theta'^*-\theta^*\|=O(t^2)$; with Proposition D the Newton error is $O(t^4)$.

*Proof.* $C_t=C_{\mathrm{old}}+tV$ with $V=\sum_jw_{vj}\frac{1-Z_vZ_j}2$. Conjugation by $X_v$ fixes $|+\rangle^{\otimes n}$, $B$ and $C_{\mathrm{old}}$
(no term of $C_{\mathrm{old}}$ contains $Z_v$), and maps $Z_vZ_j\mapsto-Z_vZ_j$, so $X_vVX_v=W_v-V$. Inserting $X_v^2=I$ everywhere,
$F_t(\theta)$ equals the objective with cost $C_{\mathrm{old}}+tW_v-tV$. The constant $tW_v$ is a global phase in the circuit and
adds $tW_v$ to the expectation, and $C_{\mathrm{old}}-tV$ is $C_{-t}$. So $F_t=tW_v+F_{-t}$. Write $F_t=tW_v/2+E_t$; then
$E_t=E_{-t}$. An even smooth function of $t$ has no linear term, and $tW_v/2$ does not depend on $\theta$. $\square$

*Meaning for the thesis.* Adding an order is a "second-order event" for the angles. This is why the observed shifts are
small (E3, E4) and why reusing the previous angles (temporal warm start) already works reasonably; it also predicts
how the shift grows with the new edge weights (checked numerically: slope 1.97 on a log-log plot, E4).
Removing a vertex is the same statement read backwards.

## Numerical check (E4, 2026-11-02)
30 complete Euclidean graphs, $n=6\to7$, $p=1$, reference optima polished to $\|\nabla F\|<10^{-12}$ with repeated Newton steps.
* Lemma D0: shift vs. $t\in\{0.05,0.1,0.2\}$, log-log slope median 1.97 (range 1.83 to 2.02). Predicted: 2.
* Proposition D: $K=\text{err}/\text{shift}^2$ stays bounded, median 2.4, max 3.7 (quadratic behaviour). Bound $M/(2\mu)$
  with the crude $M$: about 1000x larger than $K$.
* Real orders ($t=1$): shift median 0.052 rad; Newton error median 0.0042 rad; Newton/reuse error ratio median 0.08, max 0.30.
  Every Newton step was "ok" (Hessian negative definite).

## Cost and limits
* Cost: 1 gradient + $2\cdot2p$ gradients for the finite-difference Hessian (`perturb.hessian`). On hardware the
  parameter-shift rule needs $O(p^2)$ circuits: cheap for $p\le3$, expensive for large $p$ (report this).
* The step needs the curvature assumption. `newton_warm_start` returns the old angles if the Hessian is not negative
  definite, and caps the step length (status `not_concave` / `capped`).
* Residual learning (C1): the ML model predicts the remaining error $\theta'^*-\hat\theta$ instead of $\theta'^*$ itself.
  Proposition D says this residual is $O(\|\delta\|^2)$, so the target is small and well scaled.
