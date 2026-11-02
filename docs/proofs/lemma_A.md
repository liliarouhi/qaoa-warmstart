# Lemma A: stability of the QAOA objective when a vertex is added (dense regime)

> **Authorship note.** Drafted with AI assistance (Claude) on 2026-11-02 at the author's request.
> The author must check every step, rewrite it in her own words for the thesis, and disclose
> AI assistance as IUST rules require. Numerical check: `tests/test_stability.py`.
> Thesis placement: Chapter 3 (theory), referenced in Chapter 6 (stability experiments).

## Setting and notation
Graph $G=(V,E,w)$ with $|V|=n$. Cost and mixer Hamiltonians (see `docs/math_conventions.md`):
$$C=\sum_{(u,v)\in E} w_{uv}\,\frac{I-Z_uZ_v}{2},\qquad B=\sum_{j\in V}X_j .$$
For $\theta=(\gamma_1,\dots,\gamma_p,\beta_1,\dots,\beta_p)$:
$$U_C(\theta)=V_p\cdots V_1,\quad V_l=e^{-i\beta_l B}e^{-i\gamma_l C},\qquad |\psi_C(\theta)\rangle=U_C(\theta)\,|+\rangle^{\otimes n},\qquad F_C(\theta)=\langle\psi_C(\theta)|C|\psi_C(\theta)\rangle .$$
$\|\cdot\|$ is the Euclidean norm on vectors and the operator norm on matrices. For a diagonal
matrix $D$, $\|D\|=\max_z |D_{zz}|$. Write $C_{\max},C_{\min}$ for the largest and smallest cut values.

## Lemma A0 (an isolated vertex changes nothing)
Let $G^+$ be $G$ plus a vertex $x$ with no edges. Then $F_{C^+}(\theta)=F_C(\theta)$ for every $\theta$.

*Proof.* $C^+=C\otimes I_x$ and $B^+=B\otimes I_x+I\otimes X_x$. The two terms of $B^+$ commute, so
$e^{-i\beta B^+}=e^{-i\beta B}\otimes e^{-i\beta X_x}$, and $e^{-i\gamma C^+}=e^{-i\gamma C}\otimes I_x$.
Hence $|\psi_{C^+}\rangle=|\psi_C\rangle\otimes|\phi\rangle$ with $|\phi\rangle=\prod_l e^{-i\beta_l X_x}|+\rangle$, a unit vector.
Then $F_{C^+}=\langle\psi_C|C|\psi_C\rangle\,\langle\phi|\phi\rangle=F_C$. $\square$

## Lemma A (perturbation of the cost Hamiltonian)
Let $C$ and $C'=C+\Delta C$ be cost Hamiltonians on the same $n$ qubits (both diagonal in the
computational basis). Then for every $\theta$ and every real constant $c$,
$$\big|F_{C'}(\theta)-F_C(\theta)\big|\;\le\;\|\Delta C\|\Big(1+2\,\|C-cI\|\sum_{l=1}^p|\gamma_l|\Big).$$
With $c=(C_{\max}+C_{\min})/2$ this gives
$$\big|F_{C'}(\theta)-F_C(\theta)\big|\;\le\;\|\Delta C\|\Big(1+(C_{\max}-C_{\min})\sum_{l=1}^p|\gamma_l|\Big).$$

*Proof.* Write $\psi=\psi_C(\theta)$, $\psi'=\psi_{C'}(\theta)$.

**Step 1 (split).**
$$F_{C'}-F_C=\langle\psi'|\Delta C|\psi'\rangle+\big(\langle\psi'|C|\psi'\rangle-\langle\psi|C|\psi\rangle\big).$$
Since $\|\psi'\|=1$, $|\langle\psi'|\Delta C|\psi'\rangle|\le\|\Delta C\|$.

**Step 2 (difference of expectations).** For unit vectors $a,b$ and Hermitian $A$,
$$\langle a|A|a\rangle-\langle b|A|b\rangle=\langle a-b|A|a\rangle+\langle b|A|a-b\rangle$$
(expand the right side: the cross terms $\pm\langle b|A|a\rangle$ cancel). By Cauchy–Schwarz,
$|\langle a|A|a\rangle-\langle b|A|b\rangle|\le 2\|A\|\,\|a-b\|$. Because $\langle a|a\rangle=\langle b|b\rangle=1$,
replacing $A$ by $A-cI$ does not change the left side, so we may use $\|C-cI\|$ in place of $\|C\|$.
Hence the second term of Step 1 is at most $2\|C-cI\|\,\|\psi'-\psi\|$.

**Step 3 (distance between the two states).** $\|\psi'-\psi\|\le\|U_{C'}-U_C\|$ because $|+\rangle^{\otimes n}$ is a unit vector.
For products of unitaries, $\|V'_p\cdots V'_1-V_p\cdots V_1\|\le\sum_l\|V'_l-V_l\|$; by induction,
$$U'_k-U_k=V'_k\,(U'_{k-1}-U_{k-1})+(V'_k-V_k)\,U_{k-1},$$
and unitaries have norm 1. The mixer factor is the same in $V_l$ and $V'_l$, so
$$\|V'_l-V_l\|=\|e^{-i\gamma_l C'}-e^{-i\gamma_l C}\|=\|e^{-i\gamma_l C}\,(e^{-i\gamma_l\Delta C}-I)\|=\max_z\big|e^{-i\gamma_l\,\Delta C_{zz}}-1\big|\le|\gamma_l|\,\|\Delta C\|,$$
using that $C$ and $\Delta C$ are diagonal (so they commute and $e^{-i\gamma(C+\Delta C)}=e^{-i\gamma C}e^{-i\gamma\Delta C}$)
and $|e^{it}-1|\le|t|$. Summing, $\|\psi'-\psi\|\le\|\Delta C\|\sum_l|\gamma_l|$.

**Step 4 (combine).** Steps 1–3 give the first inequality. For the second, $C-cI$ is diagonal with entries
$C_{zz}-c\in[C_{\min}-c,\,C_{\max}-c]$, so $\|C-cI\|=(C_{\max}-C_{\min})/2$ for $c=(C_{\max}+C_{\min})/2$. $\square$

## Corollary A1 (new order in the dense DVRP graph)
Let $G$ have $n$ vertices and let $G'$ add a vertex $x$ joined to every old vertex with weights
$w_{xi}\ge0$, keeping the old weights unchanged. Then for every $\theta$
$$\big|F_{G'}(\theta)-F_G(\theta)\big|\le d_x\Big(1+(C_{\max}-C_{\min})\sum_l|\gamma_l|\Big),\qquad d_x=\sum_{i}w_{xi}\le n\max_i w_{xi}.$$

*Proof.* By Lemma A0, compare $G'$ with $G^+$ (same $F$ as $G$). On $n+1$ qubits,
$\Delta C=\sum_i w_{xi}(I-Z_xZ_i)/2$ is diagonal with entries $\sum_i w_{xi}[z_x\ne z_i]\in[0,d_x]$, so $\|\Delta C\|\le d_x$.
Apply Lemma A ($C_{\max},C_{\min}$ of $G^+$ equal those of $G$). $\square$

## Remarks for the thesis
1. **How tight is it?** On 200 random complete Euclidean graphs ($n=8\to9$, $p=1$, $\gamma\in[0,1]$),
   the ratio (actual change)/(bound) had median 0.07 and maximum 0.49 (script in `docs/experiments.md`, E1).
   So the bound is valid but typically about 14 times larger than the actual change. Tightening it is open.
2. **Relative size.** $F$ itself is of order $W=\sum w$, which grows like $n^2$ on a complete graph, while
   $d_x$ grows like $n$. The first term is therefore $O(1/n)$ relative to $F$. The second term is small only if
   $(C_{\max}-C_{\min})\sum|\gamma_l|$ is bounded; whether good angles scale like $1/W$ is **a conjecture to test**
   (compare Sureshbabu et al. 2024 on weight rescaling).
3. **Weight normalisation.** If weights are re-normalised by a new maximum distance, every old weight is scaled by
   $s\le1$ and $\Delta C=(s-1)C+\Delta C_{\text{new}}$, so $\|\Delta C\|\le(1-s)\|C\|+d_x$. To keep Corollary A1 clean,
   fix the scale for the whole stream (decision D8).
4. **Novelty.** The argument is a standard perturbation bound; do not claim it as new without a literature check.
   What can be new is its use for dynamic DVRP streams and the comparison with the sparse (light-cone) regime.
5. **From $F$ to the optimal angles.** Lemma A bounds the change of the objective, not of $\theta^*$. Theorem C
   (next) adds a curvature assumption at $\theta^*$ to bound $\|\theta^*(G')-\theta^*(G)\|$.
