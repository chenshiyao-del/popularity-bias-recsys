import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:           
    HAVE_NUMBA = False
    print("numba not installed -> using numpy fallback. `pip install numba` for a big speedup.")

CATEGORY = "Amazon_Fashion"

REPO_DIR = Path(__file__).resolve().parent.parent
IN_PATH = REPO_DIR / "data" / "interim" / f"{CATEGORY}_interactions.parquet"
OUT_DIR = REPO_DIR / "data" / "processed"


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
EVAL_EVERY = 5        
TOP_K = 10
N_EVAL_USERS = 10_000 
SEED = 42

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Load and map ratings to preference + confidence
# ---------------------------------------------------------------------------
df = pd.read_parquet(IN_PATH, columns=["user_id", "parent_asin", "rating", "timestamp"])
r = df["rating"].round().astype(int)
df["pref"] = r.map({k: v[0] for k, v in RATING_MAP.items()}).astype("float32")
df["conf"] = (1.0 + ALPHA * r.map({k: v[1] for k, v in RATING_MAP.items()})).astype("float32")

# integer ids for matrix rows/cols
df["u"], user_ids = pd.factorize(df["user_id"])
df["i"], item_ids = pd.factorize(df["parent_asin"])
n_users, n_items = len(user_ids), len(item_ids)
print(f"{len(df):,} interactions | {n_users:,} users | {n_items:,} items")


# ---------------------------------------------------------------------------
# 2. Temporal leave-last-out split
#    Each user with >= 2 interactions: latest *positive* (3-5 star) one -> test.
#    Users with 1 interaction stay in train only (no k-core filtering).
# ---------------------------------------------------------------------------
df = df.sort_values(["u", "timestamp"])
n_per_user = df.groupby("u")["u"].transform("size")
cand = df[(df["pref"] == 1.0) & (n_per_user >= 2)]
test_idx = cand.groupby("u").tail(1).index
test = df.loc[test_idx, ["u", "i"]]
train = df.drop(index=test_idx)
print(f"train {len(train):,} | test users {len(test):,}")

def build_csr(rows, cols, vals, shape):
    m = sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
    m.sort_indices()
    return m

shape = (n_users, n_items)
C_ui = build_csr(train["u"], train["i"], train["conf"], shape)   # confidence
P_ui = build_csr(train["u"], train["i"], train["pref"], shape)   # preference
C_iu, P_iu = C_ui.T.tocsr(), P_ui.T.tocsr()                      # item-side view
for m in (C_iu, P_iu):
    m.sort_indices()
# C and P must share the same sparsity pattern (P keeps explicit 0s for negatives)
assert C_ui.nnz == P_ui.nnz and C_iu.nnz == P_iu.nnz


# ---------------------------------------------------------------------------
# 3. ALS solver
#    For row u with observed items S_u (fixed other-side factors Y):
#      (Y^T Y + sum_{i in S_u} (c_ui - 1) y_i y_i^T + REG*I) x_u
#          = sum_{i in S_u} c_ui p_ui y_i
#    Y^T Y covers all the unrated cells (c=1, p=0) in one shot, so cost scales
#    with the number of ratings, not users x items.  Rows are solved in
#    vectorised batches instead of a Python loop over 2M users.
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
# 4. Evaluation: accuracy + popularity bias
# ---------------------------------------------------------------------------
item_pop = np.bincount(train["i"], minlength=n_items)            # train popularity
head_items = np.zeros(n_items, dtype=bool)                      # top 20% by popularity
head_items[np.argsort(-item_pop)[: max(1, n_items // 5)]] = True

eval_test = test if N_EVAL_USERS is None or len(test) <= N_EVAL_USERS \
    else test.sample(N_EVAL_USERS, random_state=SEED)

def evaluate(X, Y, k=TOP_K, batch=256):
    us, targets = eval_test["u"].to_numpy(), eval_test["i"].to_numpy()
    hits, ndcg, arp, head = [], [], [], []
    recommended = set()
    for s in range(0, len(us), batch):
        bu, bt = us[s:s + batch], targets[s:s + batch]
        scores = X[bu] @ Y.T
        seen = C_ui[bu]                                         # never re-recommend train items
        scores[np.repeat(np.arange(len(bu)), np.diff(seen.indptr)), seen.indices] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)

        match = top == bt[:, None]
        hits.append(match.any(axis=1))
        rank = match.argmax(axis=1)
        ndcg.append(np.where(match.any(axis=1), 1.0 / np.log2(rank + 2), 0.0))
        arp.append(item_pop[top].mean(axis=1))
        head.append(head_items[top].mean(axis=1))
        recommended.update(top.ravel().tolist())
    return {
        f"Recall@{k}": np.concatenate(hits).mean(),
        f"NDCG@{k}": np.concatenate(ndcg).mean(),
        "ARP": np.concatenate(arp).mean(),            # avg popularity of recs
        "HeadShare": np.concatenate(head).mean(),     # share of recs from top-20% items
        "Coverage": len(recommended) / n_items,       # catalog coverage
    }


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
X = (rng.standard_normal((n_users, FACTORS)) * 0.01).astype(np.float32)
Y = (rng.standard_normal((n_items, FACTORS)) * 0.01).astype(np.float32)

for it in range(1, ITERATIONS + 1):
    t0 = time.perf_counter()
    X = als_step(C_ui, P_ui, Y)
    t1 = time.perf_counter()
    Y = als_step(C_iu, P_iu, X)
    t2 = time.perf_counter()
    msg = f"iter {it:2d} | users {t1 - t0:.1f}s | items {t2 - t1:.1f}s"
    if it % EVAL_EVERY == 0 or it == ITERATIONS:
        m = evaluate(X, Y)
        msg += f" | eval {time.perf_counter() - t2:.1f}s | " + " | ".join(f"{k} {v:.4f}" for k, v in m.items())
    print(msg, flush=True)

# reference point: recommending the most popular items to everyone
pop_item_vec = np.log1p(item_pop).astype(np.float32)[:, None]   # score = popularity
pop_metrics = evaluate(np.ones((n_users, 1), np.float32), pop_item_vec)
print("\nPopularity baseline: " + " | ".join(f"{k} {v:.4f}" for k, v in pop_metrics.items()))

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(OUT_DIR / f"{CATEGORY}_wmf.npz", user_factors=X, item_factors=Y,
         user_ids=np.asarray(user_ids, dtype=str), item_ids=np.asarray(item_ids, dtype=str))
print("Saved factors to", OUT_DIR / f"{CATEGORY}_wmf.npz")