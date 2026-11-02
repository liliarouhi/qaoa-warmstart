PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA user_version = 3;   -- schema version (v2: split, stream, ngev, more init methods, D13; v3: feature_projection, D15)

-- 1) DVRP world (operational data)
CREATE TABLE depot    (depot_id INTEGER PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL);
CREATE TABLE vehicle  (vehicle_id INTEGER PRIMARY KEY,
                       depot_id INTEGER NOT NULL REFERENCES depot(depot_id),
                       capacity REAL NOT NULL CHECK (capacity > 0));
CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL);
CREATE TABLE orders   (order_id INTEGER PRIMARY KEY,
                       stream_id INTEGER REFERENCES stream(stream_id) ON DELETE CASCADE,
                       customer_id INTEGER NOT NULL REFERENCES customer(customer_id),
                       demand REAL NOT NULL CHECK (demand > 0),
                       arrival_t REAL NOT NULL,
                       status TEXT NOT NULL CHECK (status IN ('pending','assigned','served')));
CREATE INDEX ix_orders_status_t ON orders(status, arrival_t);

-- 2) Graph snapshots: one Max-Cut instance per re-optimization event
CREATE TABLE graph (graph_id   INTEGER PRIMARY KEY,
                    wl_hash    TEXT NOT NULL,          -- isomorphism-invariant candidate key (may collide)
                    n_nodes    INTEGER NOT NULL CHECK (n_nodes BETWEEN 2 AND 30),
                    n_edges    INTEGER NOT NULL,
                    weight_scale REAL NOT NULL,         -- normalisation factor applied to weights
                    family     TEXT NOT NULL,           -- 'ER','REG','DVRP',...
                    split      TEXT NOT NULL DEFAULT 'train' CHECK (split IN ('train','val','test')));
CREATE INDEX ix_graph_hash ON graph(wl_hash);
CREATE INDEX ix_graph_split ON graph(split, family, n_nodes);
CREATE TABLE graph_edge (graph_id INTEGER NOT NULL REFERENCES graph(graph_id) ON DELETE CASCADE,
                         u INTEGER NOT NULL, v INTEGER NOT NULL, w REAL NOT NULL,
                         PRIMARY KEY (graph_id, u, v), CHECK (u < v)) WITHOUT ROWID;
CREATE TABLE stream (stream_id INTEGER PRIMARY KEY, seed INTEGER NOT NULL, config TEXT NOT NULL);  -- config = JSON
CREATE TABLE snapshot (snapshot_id INTEGER PRIMARY KEY,
                       stream_id INTEGER NOT NULL REFERENCES stream(stream_id) ON DELETE CASCADE,
                       seq INTEGER NOT NULL,                -- position in the stream (0,1,2,...)
                       event TEXT NOT NULL CHECK (event IN ('add','remove')),
                       prev_ok INTEGER NOT NULL CHECK (prev_ok IN (0,1)),   -- seq-1 is the state just before
                       t REAL NOT NULL,
                       graph_id INTEGER NOT NULL REFERENCES graph(graph_id),
                       trigger_order INTEGER REFERENCES orders(order_id),
                       UNIQUE (stream_id, seq));
CREATE TABLE snapshot_node (snapshot_id INTEGER NOT NULL REFERENCES snapshot(snapshot_id) ON DELETE CASCADE,
                            node_idx INTEGER NOT NULL,
                            order_id INTEGER NOT NULL REFERENCES orders(order_id),
                            PRIMARY KEY (snapshot_id, node_idx)) WITHOUT ROWID;

-- 3) Features: 1:1 with graph (feature depends only on graph -> separate table, 3NF)
CREATE TABLE graph_feature (graph_id INTEGER PRIMARY KEY REFERENCES graph(graph_id) ON DELETE CASCADE,
                            feat_version INTEGER NOT NULL,
                            vec BLOB NOT NULL);     -- full float32 vector
CREATE VIRTUAL TABLE feature_rtree USING rtree(graph_id, f1_lo, f1_hi, f2_lo, f2_hi, f3_lo, f3_hi);  -- PCA-3 projection
-- PCA-3 projection used to fill feature_rtree (single row; see qws/db/similarity.py, decision D15)
CREATE TABLE feature_projection (
    proj_id      INTEGER PRIMARY KEY CHECK (proj_id = 1),
    feat_version INTEGER NOT NULL,
    dim          INTEGER NOT NULL CHECK (dim = 3),
    n_fit        INTEGER NOT NULL,
    explained    REAL NOT NULL,
    mu BLOB NOT NULL, sd BLOB NOT NULL, comps BLOB NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP);

-- 4) QAOA experiments
CREATE TABLE qaoa_run (run_id INTEGER PRIMARY KEY,
                       graph_id INTEGER NOT NULL REFERENCES graph(graph_id) ON DELETE CASCADE,
                       p INTEGER NOT NULL CHECK (p >= 1),
                       init_method TEXT NOT NULL CHECK (init_method IN ('random','fixed','interp','tqa','temporal','newton','multistart','ml','cache','ml+cache')),
                       optimizer TEXT NOT NULL, seed INTEGER NOT NULL,
                       nfev INTEGER NOT NULL, ngev INTEGER NOT NULL DEFAULT 0, energy REAL NOT NULL, ratio REAL CHECK (ratio BETWEEN 0 AND 1.000001),
                       wall_ms REAL NOT NULL, model_id INTEGER REFERENCES model(model_id),
                       UNIQUE (graph_id, p, init_method, optimizer, seed));
CREATE INDEX ix_run_lookup ON qaoa_run(graph_id, p, energy DESC);
CREATE TABLE qaoa_param (run_id INTEGER NOT NULL REFERENCES qaoa_run(run_id) ON DELETE CASCADE,
                         layer INTEGER NOT NULL,
                         stage TEXT NOT NULL CHECK (stage IN ('init','final')),
                         gamma REAL NOT NULL, beta REAL NOT NULL,
                         PRIMARY KEY (run_id, stage, layer)) WITHOUT ROWID;

-- 5) Cache = best known canonical parameters per (graph, p)
CREATE TABLE param_cache (graph_id INTEGER NOT NULL REFERENCES graph(graph_id) ON DELETE CASCADE,
                          p INTEGER NOT NULL,
                          best_run_id INTEGER NOT NULL REFERENCES qaoa_run(run_id),
                          best_energy REAL NOT NULL,
                          hits INTEGER NOT NULL DEFAULT 0,
                          last_used REAL,
                          PRIMARY KEY (graph_id, p));
CREATE TABLE model (model_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, feat_version INTEGER NOT NULL,
                    train_sql TEXT NOT NULL, val_mse REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, artifact_path TEXT NOT NULL);

-- keep cache consistent automatically
CREATE TRIGGER trg_cache_upsert AFTER INSERT ON qaoa_run
BEGIN
  INSERT INTO param_cache(graph_id, p, best_run_id, best_energy)
  VALUES (NEW.graph_id, NEW.p, NEW.run_id, NEW.energy)
  ON CONFLICT(graph_id, p) DO UPDATE SET best_run_id = excluded.best_run_id, best_energy = excluded.best_energy
  WHERE excluded.best_energy > param_cache.best_energy;
END;
