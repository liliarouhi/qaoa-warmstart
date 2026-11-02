# Database design notes

Schema: `src/qws/db/schema.sql` (thesis Appendix A). DAO: `src/qws/db/repository.py`.

## Entities (13 tables + 1 virtual)
- Operational DVRP world: `depot`, `vehicle`, `customer`, `orders`
- Max-Cut instances: `graph`, `graph_edge`, `snapshot`, `snapshot_node`
- Learning and experiments: `graph_feature`, `qaoa_run`, `qaoa_param`, `param_cache`, `model`
- Similarity index: `feature_rtree` (virtual R*Tree on a 3-D PCA projection of features)

Task for the author: draw the ER diagram (draw.io or dbdiagram.io), list cardinalities.

## Functional dependencies (starting list, to be completed by the author)
| Table | Key | FDs |
|---|---|---|
| graph | graph_id | graph_id -> wl_hash, n_nodes, n_edges, weight_scale, family |
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
| `feature_rtree` | k-NN by bounding-box filter, then exact distance on candidates |

Phase 3: record `EXPLAIN QUERY PLAN` before/after each index for Chapter 5.

## Transactions and concurrency
`record_run` writes the run and all its parameters inside `BEGIN IMMEDIATE ... COMMIT`;
any error rolls back everything (tested). WAL mode lets the online loop read while the
offline trainer writes.
