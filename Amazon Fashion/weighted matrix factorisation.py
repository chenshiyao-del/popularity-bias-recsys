"""
wmf.py -- Weighted Matrix Factorization (WMF). Model code only.

Data  : loaded through leakage_guard.guard_all from config.DATA_DIR, the folder
        split.py saved into. No splitting happens here.
Files : config.py, split.py, leakage_guard.py, rec_eval.py and this file sit in src/.
Eval  : rec_eval.evaluate (shared by MostPopular, ItemCF, WMF, NCF).

Model (Hu, Koren & Volinsky 2008, extended with explicit negatives):
  every (user, item) cell has a preference p and a confidence c
    rated 3-5  -> p = 1, c = 1 + ALPHA * w   (w: 5->1.0, 4->0.75, 3->0.5)
    rated 1-2  -> p = 0, c = 1 + ALPHA * w   (w: 1->1.0, 2->0.5)  "confident dislike"
    unrated    -> p = 0, c = 1               "weak, uncertain negative"
  minimise  sum_ui c_ui (p_ui - x_u . y_i)^2 + REG * (|X|^2 + |Y|^2), solved with ALS.

Item columns use the shared popularity-rank index (stats.rank_of), so the
recommendations are already in the id space rec_eval expects.
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

import config
from leakage_guard import guard_all
from rec_eval import evaluate, sample_test_users, save_metrics

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:           # falls back to a (slower) chunked numpy solver
    HAVE_NUMBA = False
    print("numba not installed -> using numpy fallback. `pip install numba` for a big speedup.")


CATEGORY = config.CATEGORY
PROC = config.DATA_DIR        

RATING_MAP = {
    5: (1.0, 1.00),
    4: (1.0, 0.75),
    3: (1.0, 0.50),
    2: (0.0, 0.50),   
    1: (0.0, 1.00),   
}

FACTORS = 32         
REG = 0.1
ALPHA = 20.0          
ITERATIONS = 10       
MAX_BATCH_NNZ = 20_000  
TOP_K = config.K
EVAL_SAMPLE = None    
SEED = 42

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Load the shared split (no re-splitting here)
# ---------------------------------------------------------------------------
train, test, stats = guard_all(PROC, CATEGORY)
if EVAL_SAMPLE is not None:
    test, _ = sample_test_users(test, n=EVAL_SAMPLE, seed=SEED)

n_items = stats.n_catalog
print(f"train {len(train):,} rows | test users {len(test):,} | catalog {n_items:,} items")


# ---------------------------------------------------------------------------
# 2. Ratings -> preference + confidence, and the sparse matrices
#    rows = users (indexed from train), cols = items (shared popularity rank)
# ---------------------------------------------------------------------------
r = train["rating"].round().astype(int)
pref = r.map({k: v[0] for k, v in RATING_MAP.items()}).to_numpy(np.float32)
conf = (1.0 + ALPHA * r.map({k: v[1] for k, v in RATING_MAP.items()})).to_numpy(np.float32)

u_idx, user_ids = pd.factorize(train["user_id"])
i_idx = train["parent_asin"].map(stats.rank_of).to_numpy()
if np.isnan(i_idx.astype(float)).any():
    raise ValueError("Some training items are missing from stats.rank_of")
i_idx = i_idx.astype(np.int64)
n_users = len(user_ids)

def build_csr(rows, cols, vals, shape):
    m = sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
    m.sort_indices()
    return m

shape = (n_users, n_items)
C_ui = build_csr(u_idx, i_idx, conf, shape)   # confidence
P_ui = build_csr(u_idx, i_idx, pref, shape)   # preference
C_iu, P_iu = C_ui.T.tocsr(), P_ui.T.tocsr()   # item-side view
for m in (C_iu, P_iu):
    m.sort_indices()
# C and P must share the same sparsity pattern (P keeps explicit 0s for negatives)
assert C_ui.nnz == P_ui.nnz and C_iu.nnz == P_iu.nnz


# ---------------------------------------------------------------------------
# 3. ALS solver
#    For row u with observed items S_u (fixed other-side factors Y):
#      (Y^T Y + sum_{i in S_u} (c_ui - 1) y_i y_i^T + REG*I) x_u
#          = sum_{i in S_u} c_ui p_ui y_i
#    Y^T Y covers all the unrated cells (c=1, p=0) in one shot.
# ---------------------------------------------------------------------------
if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _solve_rows(indptr, indices, cdata, pdata, Y, base, X):
        n, k = X.shape
        for u in prange(n):
            s, e = indptr[u], indptr[u + 1]
            if e == s:
                continue                      # no ratings -> factor stays 0
            Yo = Y[indices[s:e]]              # (m, k)
            c = cdata[s:e]
            Yw = Yo * (c - np.float32(1.0)).reshape(-1, 1)
            A = base + np.ascontiguousarray(Yw.T) @ Yo
            b = ((c * pdata[s:e]).reshape(1, -1) @ Yo).reshape(-1)
            X[u] = np.linalg.solve(A, b)

    def als_step(C, P, Y):
        k = Y.shape[1]
        base = (Y.T @ Y + REG * np.eye(k, dtype=np.float32)).astype(np.float32)
        X = np.zeros((C.shape[0], k), dtype=np.float32)
        _solve_rows(C.indptr, C.indices, C.data, P.data, Y, base, X)
        return X
else:
    def als_step(C, P, Y):
        n, k = C.shape[0], Y.shape[1]
        base = Y.T @ Y + REG * np.eye(k, dtype=np.float32)
        X = np.zeros((n, k), dtype=np.float32)
        indptr = C.indptr
        start = 0
        while start < n:
            # batch by number of ratings (not rows) so memory stays bounded
            stop = int(np.searchsorted(indptr, indptr[start] + MAX_BATCH_NNZ, side="right")) - 1
            stop = min(max(stop, start + 1), n)
            lo, hi = indptr[start], indptr[stop]
            if hi == lo:
                start = stop
                continue
            has = np.diff(indptr[start:stop + 1]) > 0
            cols, c, p = C.indices[lo:hi], C.data[lo:hi], P.data[lo:hi]
            Yo = Y[cols]
            Yw = Yo * (c - 1.0)[:, None]
            if stop - start == 1:             # single (possibly huge) row: plain matmul
                A = (base + Yw.T @ Yo)[None]
                b = ((c * p)[:, None] * Yo).sum(axis=0, keepdims=True)
            else:
                outer = np.einsum("ni,nj->nij", Yw, Yo)   # one temp, not two
                seg = indptr[start:stop][has] - lo
                A = base + np.add.reduceat(outer, seg, axis=0)
                b = np.add.reduceat((c * p)[:, None] * Yo, seg, axis=0)
            X[start:stop][has] = np.linalg.solve(A, b[..., None])[..., 0]
            start = stop
        return X


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
X = (rng.standard_normal((n_users, FACTORS)) * 0.01).astype(np.float32)
Y = (rng.standard_normal((n_items, FACTORS)) * 0.01).astype(np.float32)

for it in range(1, ITERATIONS + 1):
    t0 = time.perf_counter()
    X = als_step(C_ui, P_ui, Y)
    t1 = time.perf_counter()
    Y = als_step(C_iu, P_iu, X)
    print(f"iter {it:2d} | users {t1 - t0:.1f}s | items {time.perf_counter() - t1:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 5. Recommend top-K for each test user (row i = test.iloc[i])
#    - already-seen training items are excluded (rec_eval checks this)
#    - catalog items with no training ratings are excluded (zero vectors would
#      otherwise score 0 and can outrank real items for users with negative scores)
#    - a test user with no training rows gets the most popular items (ranks 0..K-1)
# ---------------------------------------------------------------------------
def recommend(X, Y, test, k=TOP_K):
    user_row = pd.Series(np.arange(n_users), index=user_ids)
    rows = test["user_id"].map(user_row).fillna(-1).to_numpy(np.int64)
    recs = np.tile(np.arange(k, dtype=np.int64), (len(test), 1))   # popularity fallback

    no_train = np.asarray(stats.items["cnt"])[:n_items] == 0
    known = np.flatnonzero(rows >= 0)
    batch = max(16, 50_000_000 // n_items)                         # ~200 MB of scores per batch
    for s in range(0, len(known), batch):
        pos = known[s:s + batch]
        bu = rows[pos]
        scores = X[bu] @ Y.T
        scores[:, no_train] = -np.inf
        seen = C_ui[bu]
        scores[np.repeat(np.arange(len(bu)), np.diff(seen.indptr)), seen.indices] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        recs[pos] = np.take_along_axis(top, order, axis=1)
    if (rows < 0).any():
        print(f"{(rows < 0).sum():,} test users had no training rows -> popularity fallback")
    return recs

t0 = time.perf_counter()
recs = recommend(X, Y, test)
print(f"recommendations built in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# 6. Shared evaluation + save
# ---------------------------------------------------------------------------
metrics = evaluate(recs, test, train, stats, K=TOP_K, model_name="WMF")
for name, value in metrics.items():
    print(f"{name:>32}: {value:.4f}" if isinstance(value, float) else f"{name:>32}: {value}")
print("Saved metrics to", save_metrics(metrics, PROC, CATEGORY))

item_ids = stats.rank_of.sort_values().index.to_numpy(dtype=str)   # position = popularity rank
np.savez(PROC / f"{CATEGORY}_wmf.npz", user_factors=X, item_factors=Y,
         user_ids=np.asarray(user_ids, dtype=str), item_ids=item_ids)
print("Saved factors to", PROC / f"{CATEGORY}_wmf.npz")