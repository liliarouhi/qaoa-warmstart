# Database design notes

Schema: `src/qws/db/schema.sql` (thesis Appendix A). DAO: `src/qws/db/repository.py`.

## Entities (15 tables + 1 virtual), schema v3
- Operational DVRP world: `depot`, `vehicle`, `customer`, `orders`, `stream` (one simulated order stream; config as JSON)
- Max-Cut instances: `graph`, `graph_edge`, `snapshot`, `snapshot_node`
- Learning and experiments: `graph_feature`, `qaoa_run`, `qaoa_param`, `param_cache`, `model`
- Similarity index: `feature_rtree` (virtual R*Tree on a 3-D PCA projection of features) and
  `feature_projection` (the stored projection: mean, std, 3 components; single row)

Task for the author: draw the ER diagram (draw.io or dbdiagram.io), list cardinalities.

## Functional dependencies (starting list, to be completed by the author)
| Table | Key | FDs |
|---|---|---|
| graph | graph_id | graph_id -> wl_hash, n_nodes, n_edges, weight_scale, family, split |
| snapshot | snapshot_id; also (stream_id, seq) | snapshot_id -> stream_id, seq, event, prev_ok, t, graph_id, trigger_order |
| graph_edge | (graph_id, u, v) | (graph_id, u, v) -> w |
| graph_feature | graph_id | graph_id -> feat_version, vec |
| qaoa_run | run_id; also (graph_id, p, init_method, optimizer, seed) | run_id -> all other columns |
| qaoa_param | (run_id, stage, layer) | (run_id, stage, layer) -> gamma, beta |
| param_cache | (graph_id, p) | (graph_id, p) -> best_run_id, best_energy, hits, last_used |

## Normalisation arguments
- **1NF:** angles are one row per layer in `qaoa_param`, not a JSON array in `qaoa_run`.
- **3NF:** features depend only on `graph_id`. Storing them in `qaoa_run` would give
  run_id -> graph_id -> features (transitive dependency). Hence the 1:1 table `graph_feature`.
- **BCNF check (to do):** for each table, every non-trivial FD's left side must be a superkey.
  `qaoa_run` has two candidate keys; both are superkeys, so it is fine.
- **Conscious redundancy:** `param_cache.best_energy` duplicates `qaoa_run.energy` of
  `best_run_id`. It is kept for O(1) cache lookups and maintained by trigger `trg_cache_upsert`.
  Write this as a justified denormalisation in the thesis.
- `graph.n_edges` is derivable from `graph_edge` (count); same justification.

## Integrity constraints
FK with `ON DELETE CASCADE`; CHECK on `p >= 1`, order status, `ratio` range, `n_nodes` range;
UNIQUE run key prevents duplicate experiments. Tested in `tests/test_db.py`.
Note: SQLite enforces FKs only with `PRAGMA foreign_keys=ON` per connection (`connect()` does it).

## Indexes
| Index | Query it serves |
|---|---|
| `ix_graph_hash(wl_hash)` | exact cache candidate lookup |
| `ix_run_lookup(graph_id, p, energy DESC)` | best run per (graph, p) |
| `ix_orders_status_t(status, arrival_t)` | pending orders in arrival order |
| `ix_graph_split(split, family, n_nodes)` | 'all unlabelled train graphs of a family, smallest first' (label.py) |
| `feature_rtree` | k-NN by cube filter in PCA-3 space, then exact 16-D distance on candidates (exact, see below) |

Phase 3: record `EXPLAIN QUERY PLAN` before/after each index for Chapter 5.

## Transactions and concurrency
`record_run` writes the run and all its parameters inside `BEGIN IMMEDIATE ... COMMIT`;
any error rolls back everything (tested). WAL mode lets the online loop read while the
offline trainer writes.

## Similarity search with the R*Tree (Phase 3, D15, E13)
Code: `src/qws/db/similarity.py`. Tests: `tests/test_similarity.py` (answers equal brute force).
- **Why a projection:** an R*Tree indexes few dimensions well; features have 16. We store the PCA-3 projection
  y = V z of the standardised features z (V has orthonormal rows). PCA-3 keeps about 87% of the variance (E13).
- **Lemma (no false negatives).** For an orthogonal projection, ||V z - V z'|| <= ||z - z'||. So every stored graph
  with full distance <= r lies in the cube of half-width r around V z. Proof: Pythagoras (the dropped components only
  add a non-negative term). The cube filter therefore never misses a true neighbour; exact distances are computed
  on the candidates only.
- **Algorithm:** grow r until the cube holds k candidates; if the k-th exact distance d_k <= r, stop; otherwise set
  r = d_k and query once more (then it always stops). Result = exact k-NN.
- **float32:** SQLite stores R*Tree coordinates as float32 and rounds box bounds outward, so the filter stays safe.
- **Query plan** (`results/rtree_bench.txt`): `SCAN r VIRTUAL TABLE INDEX 2:D1B0D3B2D5B4` (the R*Tree answers the six
  range constraints) followed by `SEARCH c USING COVERING INDEX sqlite_autoindex_param_cache_1 (graph_id=? AND p=?)`.
  The full scan without the index is `SCAN graph_feature`.
- **Limitation (honest):** the speed-up (2-15x, E13) is versus reading every vector from disk; versus vectors already
  in RAM it is only faster at N = 100000 (10.8 vs 14.5 ms) and slower for small N. For the thesis data (about 1000 graphs) the in-memory cache is
  enough; the index matters when the cache keeps growing (C3) and does not fit in memory, or many processes share it.
