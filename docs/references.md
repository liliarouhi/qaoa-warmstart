# References

**Canonical bibliography:** `docs/references.bib` is a copy of the project's Crossref-verified
bibliography (`/mnt/project-files/references/references.bib`, ~55 entries, annotated in Persian in
`references_annotated_fa.md`). Update the project copy first, then re-copy it here.

Three verified entries were added at the end of the file as a supplement (DOIs checked 2026-11-02):
- Sureshbabu et al., *Parameter Setting in Quantum Approximate Optimization of Weighted Problems*, Quantum 8, 1231 (2024). doi:10.22331/q-2024-01-18-1231
- Amosy et al., *Iteration-free quantum approximate optimization algorithm using neural networks*, Quantum Machine Intelligence 6, 38 (2024). doi:10.1007/s42484-024-00159-y
- Abbas et al., *Challenges and opportunities in quantum optimization*, Nature Reviews Physics 6, 718-735 (2024). doi:10.1038/s42254-024-00770-9

## Closest prior work (affects novelty claims)
| Our component | Prior work to position against | Consequence for the thesis |
|---|---|---|
| Lemma 2 (symmetries) | Shaydulin, Hadfield, Hogg, Safro 2021 (`shaydulin2021symmetries`) | The symmetries are known; our contribution is using them to canonicalise *regression labels* and measuring the effect (RQ2). |
| Weighted Max-Cut parameter transfer | Shaydulin et al. 2023 (`shaydulin2023weighted`), Sureshbabu et al. 2024 | Fixed/transferred parameters are a strong baseline; ML must beat them (D4 baseline). |
| Similarity-based cache | Galda et al. 2021, 2023 (`galda2021transfer`, `galda2023similarity`) | Similarity-based transfer exists; our novelty must be the DB-backed online cache in a dynamic DVRP stream, not similarity transfer itself. |
| Learned initialisation | Jain et al. 2022 (GNN), Amosy et al. 2024 | ML warm-starts exist; we compare with theory-motivated features and a fixed-parameter baseline. |
