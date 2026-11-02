# Lemma B: locality (light cone) of the QAOA objective (sparse regime)

> **Authorship note.** Drafted with AI assistance (Claude) on 2026-11-02 at the author's request.
> The author must check every step, rewrite it in her own words, and disclose AI assistance as IUST
> rules require. Numerical check: `tests/test_locality.py`. Thesis placement: Chapter 3.
> The light-cone property itself is known (Farhi, Goldstone, Gutmann 2014, arXiv:1411.4028);
> Corollaries B1-B2 apply it to the dynamic setting.

## Notation
As in `lemma_A.md`. For an edge $e=(u,v)$ write $C_e=w_e\,(I-Z_uZ_v)/2$, so $F_C=\sum_e\langle C_e\rangle$.
$\operatorname{dist}(x,S)$ is the graph distance from vertex $x$ to the set $S$. The *support*
$\operatorname{supp}(O)$ of an operator is the set of qubits it acts on non-trivially.
$N[S]=S\cup\{y:\ y \text{ adjacent to } S\}$ is the closed neighbourhood.

## Lemma B
Fix an edge $e=(u,v)$ and depth $p$. Let $E_e^{(p)}$ be the set of edges with at least one endpoint
at distance $\le p-1$ from $\{u,v\}$. Then $\langle\psi(\theta)|C_e|\psi(\theta)\rangle$ depends only on the
weights of the edges in $E_e^{(p)}$: replacing $G$ by the graph with edge set $E_e^{(p)}$ (same vertices)
does not change it, for every $\theta$.

*Proof.* Heisenberg picture: $\langle C_e\rangle=\langle+|\,O_0\,|+\rangle$ with
$O_p=C_e$ and $O_{l-1}=V_l^\dagger O_l V_l$, $V_l=e^{-i\beta_lB}e^{-i\gamma_lC}$, for $l=p,\dots,1$.

*Claim:* $\operatorname{supp}(O_{l-1})\subseteq N[S_l]$ where $S_l=\operatorname{supp}(O_l)$, and $O_{l-1}$ depends only
on weights of edges touching $S_l$.

(a) Mixer. $e^{i\beta B}=\bigotimes_j e^{i\beta X_j}$ is a product of one-qubit unitaries. Factors on qubits
outside $S_l$ commute with $O_l$ and cancel with their inverses, so $e^{i\beta B}O_le^{-i\beta B}$ is still
supported on $S_l$.

(b) Phase. Split $C=C_{\text{in}}+C_{\text{out}}$, where $C_{\text{in}}$ collects the edges with at least one endpoint
in $S_l$ and $C_{\text{out}}$ the rest. $C_{\text{out}}$ acts only outside $S_l$, so it commutes with the (mixed)
operator; it also commutes with $C_{\text{in}}$ (all terms are diagonal). Hence
$e^{i\gamma C}Pe^{-i\gamma C}=e^{i\gamma C_{\text{in}}}Pe^{-i\gamma C_{\text{in}}}$, which is supported on
$\operatorname{supp}(C_{\text{in}})\cup S_l\subseteq N[S_l]$ and involves only edges touching $S_l$.

(Order inside $V_l^\dagger O V_l=e^{i\gamma C}e^{i\beta B}Oe^{-i\beta B}e^{-i\gamma C}$: first (a), then (b).)

Starting from $S_p=\{u,v\}$ and applying the claim $p$ times, the layer $l$ step uses only edges touching
$S_l\subseteq\{y:\operatorname{dist}(y,\{u,v\})\le p-l\}$. Over all layers these are edges with an endpoint at
distance $\le p-1$, i.e. $E_e^{(p)}$. Finally $|+\rangle^{\otimes n}$ is a product state, so
$\langle+|O_0|+\rangle$ depends only on $O_0$. $\square$

## Corollary B1 (a new order far away changes nothing)
Add a vertex $x$ with any edges. If $\operatorname{dist}_{G'}(x,\{u,v\})\ge p+1$, then $\langle C_e\rangle$ is unchanged
for every $\theta$.

*Proof.* A new edge $(x,i)$ is in $E_e^{(p)}$ only if $x$ or $i$ is within $p-1$ of $\{u,v\}$, which forces
$\operatorname{dist}(x,\{u,v\})\le p$. Distances among vertices within $p-1$ of $\{u,v\}$ are not shortened by $x$
(a path through $x$ has length $\ge p+1$ to reach $x$). So $E_e^{(p)}$ is the same in $G$ and $G'$; by Lemma A0
the extra qubit does not matter. $\square$
The threshold is sharp: at distance exactly $p$ the term changes in general (tested on a path graph).

## Corollary B2 (bound with bounded degree)
Let all weights be non-negative and let $A$ be the old edges with $\operatorname{dist}(x,e)\le p$, plus the new edges.
Then
$$|F_{G'}(\theta)-F_G(\theta)|\le\sum_{e\in A}w_e .$$
If every vertex has degree $\le D$, then $|A|\le D\,|B_p(x)|\le D\big(1+D\sum_{k=0}^{p-1}(D-1)^k\big)=O(D^{p+1})$,
**independent of $n$**.

*Proof.* Terms outside $A$ are unchanged (B1). Each $\langle C_e\rangle\in[0,w_e]$ because $0\le C_e\le w_eI$, so each
changed (or new) term moves by at most $w_e$. Ball size: at most $D(D-1)^{k-1}$ vertices at distance $k\ge1$. $\square$

## Remarks for the thesis
1. **Sparse vs dense.** For bounded degree the change is $O(D^{p+1})$ while $F$ grows like $m$, so the relative
   change is $O(1/n)$ *and only local terms move*. On the complete DVRP graph $A$ is all edges and B2 reduces to
   the trivial bound; Lemma A is the right tool there. This contrast is a result to report in Chapter 6.
2. Lemma B is also what makes graph features local (degrees, triangles) meaningful for $p=1$ (compare Lemma 3).
