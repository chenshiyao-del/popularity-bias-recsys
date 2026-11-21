"""Item-based CF on the FULL catalog, evaluated through the shared pipeline (same split, users and metrics as MostPopular)."""
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from leakage_guard import guard_all
from rec_eval import evaluate, sample_test_users, save_metrics

# ---------- Configuration ----------
PROC = Path(r"D:\CS4100\group project\popularity-bias-recsys\data\interim\processed")
CATEGORY = "Beauty_and_Personal_Care"
K = 10
N_EVAL = 100_000                  # evaluation users (use the SAME n and seed for every model, e.g. 100_000 / 42)
SEED = 42
MIN_ITEM = 1                    # 1 = FULL catalog (every item seen in train); 5 = only items with >= 5 train interactions
CHUNK = 100                     # users scored per batch (memory ~ CHUNK x n_catalog x 12 bytes; lower it if RAM is tight)
MODEL_NAME = "ItemCF_full" if MIN_ITEM == 1 else f"ItemCF_min{MIN_ITEM}"

# ---------- 1. Shared data + leakage checks ----------
train, test, stats = guard_all(PROC, CATEGORY)
sub, idx = sample_test_users(test, n=N_EVAL, seed=SEED)     # fixed evaluation users shared by all models

# ---------- 2. Item index = popularity rank (shared index space) ----------
# Ranks are sorted by train popularity, so "items with >= MIN_ITEM interactions" is exactly ranks [0, n_keep).
n_keep = int((stats.items["cnt"] >= MIN_ITEM).sum())
print(f"CF catalog: {n_keep:,} of {stats.n_catalog:,} items (>= {MIN_ITEM} train interactions)")

tr_rank = train["parent_asin"].map(stats.rank_of).to_numpy()
u_codes, u_uniques = pd.factorize(train["user_id"])
m = tr_rank < n_keep
R = csr_matrix((np.ones(m.sum(), dtype="float32"), (u_codes[m], tr_rank[m])),
               shape=(len(u_uniques), n_keep))
user_row = pd.Series(np.arange(len(u_uniques)), index=u_uniques)

# ---------- 3. Item-item cosine similarity ----------
sim = cosine_similarity(R.T, dense_output=False).tocsr()
sim.setdiag(0)
sim.eliminate_zeros()
print(f"similarity matrix: {sim.nnz:,} non-zeros")

# ---------- 4. Recommend for the shared evaluation users ----------
# Zero-score items are ordered by popularity (a MostPopular fallback) through a tiny rank-based bonus.
tie_break = (1e-6 * (1 - np.arange(n_keep) / n_keep)).astype("float32")

def topk_exact(scores, k=K):
    """Top-k per row: score descending; exact ties -> more popular item (lower rank) first. Deterministic."""
    kth = np.partition(scores, -k, axis=1)[:, -k]            # k-th largest score of each row
    r, c = np.nonzero(scores >= kth[:, None])                # row-major => columns (= popularity ranks) ascending
    order = np.lexsort((c, -scores[r, c], r))                # by row, then score desc, then rank asc
    r, c = r[order], c[order]
    starts = np.searchsorted(r, np.arange(scores.shape[0]))
    return c[starts[:, None] + np.arange(k)[None, :]]

def recommend(rows, k=K):
    seen = R[rows]                                           # sparse (chunk x n_catalog)
    scores = (seen @ sim).toarray()                          # dense float32
    scores += tie_break
    r, c = seen.nonzero()
    scores[r, c] = -np.inf                                   # never re-recommend items from the user's train history
    return topk_exact(scores, k)

rows = user_row.loc[sub["user_id"]].to_numpy()
recs = np.vstack([recommend(rows[i:i + CHUNK]) for i in range(0, len(rows), CHUNK)])
print(f"recommendations done: {recs.shape}")

# ---------- 5. Shared evaluation ----------
metrics = evaluate(recs, sub, train, stats, K=K, model_name=MODEL_NAME)
for k_, v in metrics.items():
    print(f"{k_:32s} {v}")
save_metrics(metrics, PROC, CATEGORY)