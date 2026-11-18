import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # repo root / src

from leakage_guard import guard_all 

# ---------- Configuration ----------
# change proc path and category here 
PROC = Path(r"D:\CS4100\group project\popularity-bias-recsys\data\interim\processed")
CATEGORY = "Beauty_and_Personal_Care"
K = 10                          # length of each recommendation list
D = 100                         # candidate depth: recs are searched among the top-D popular items
CHUNK = 500_000                 # users processed per batch (keeps memory low)

# ---------- 1. Load data + run all leakage checks ----------
# train/test are tagged by load_split(); stats is built from train only.
train, test, stats = guard_all(PROC, CATEGORY)

# ---------- 2. "Fit": popularity ranking comes from train-only statistics ----------
pop = stats.items                                   # columns: parent_asin, cnt, mean_rating; row index = rank
rank_of = stats.rank_of                             # item -> popularity rank (0 = most popular)
cnt_by_rank = pop["cnt"].to_numpy()
is_head_by_rank = stats.is_head_by_rank
n_catalog = stats.n_catalog
print(f"catalog items: {n_catalog:,} | test users: {len(test):,}")

# ---------- 3. Encode users/items as integers ----------
user_code = pd.Series(np.arange(len(test)), index=test["user_id"])    # test user -> row number
t_rank = test["parent_asin"].map(rank_of).to_numpy()                  # popularity rank of each test item

# For every training interaction with a top-D item, record (user row, item rank).
# Only these can push an item out of a user's top-K list.
tr_rank = train["parent_asin"].map(rank_of).to_numpy()
tr_user = train["user_id"].map(user_code)
keep = (tr_rank < D) & tr_user.notna().to_numpy()
su = tr_user[keep].to_numpy().astype(np.int64)
sr = tr_rank[keep].astype(np.int64)
order = np.argsort(su, kind="stable")
su, sr = su[order], sr[order]

# ---------- 4. Generate Top-K lists in batches ----------
n = len(test)
exposure = np.zeros(D, dtype=np.int64)          # how many times each top-D item is recommended
recs_all = np.empty((n, K), dtype=np.int32)     # recommended item ranks per test user
hit = np.zeros(n, dtype=bool)
ndcg = np.zeros(n)

for a in range(0, n, CHUNK):
    b = min(a + CHUNK, n)
    seen = np.zeros((b - a, D), dtype=bool)                 # seen[u, j]: user u already interacted with rank-j item
    lo, hi = np.searchsorted(su, [a, b])
    seen[su[lo:hi] - a, sr[lo:hi]] = True

    unseen = ~seen                                          # never recommend items from the user's training history
    cum = np.cumsum(unseen, axis=1, dtype=np.int16)
    rec_mask = unseen & (cum <= K)                          # the first K unseen items in popularity order
    assert (rec_mask.sum(1) == K).all(), "A user has <K unseen items in top-D; increase D."

    _, cols = np.nonzero(rec_mask)                          # row-major order => ranked list per user
    recs = cols.reshape(-1, K)
    recs_all[a:b] = recs
    exposure += rec_mask.sum(0)

    # Accuracy: one held-out item per user, so Recall@K equals HitRate@K.
    match = recs == t_rank[a:b, None]
    h = match.any(1)
    pos = match.argmax(1)                                   # 0-based position of the test item in the list
    hit[a:b] = h
    ndcg[a:b] = np.where(h, 1.0 / np.log2(pos + 2), 0.0)
    print(f"  processed {b:,}/{n:,} users", end="\r")
print()

# ---------- 5. Metrics ----------
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n_, cum_ = len(x), np.cumsum(x)
    return (n_ + 1 - 2 * (cum_ / cum_[-1]).sum()) / n_

total_slots = n * K
full_exposure = np.zeros(n_catalog)
full_exposure[:D] = exposure                                 # items outside top-D are never recommended

test_is_head = is_head_by_rank[t_rank]
metrics = {
    "model": "MostPopular",
    "K": K,
    "n_users": int(n),
    # accuracy
    "Recall@10": float(hit.mean()),
    "NDCG@10": float(ndcg.mean()),
    "Recall@10_head_test_items": float(hit[test_is_head].mean()),
    "Recall@10_tail_test_items": float(hit[~test_is_head].mean()),
    "share_of_test_items_in_head": float(test_is_head.mean()),
    # bias
    "ARP": float((exposure * cnt_by_rank[:D]).sum() / total_slots),   # avg training popularity of recommended items
    "Gini_exposure": float(gini(full_exposure)),
    "Coverage": float((full_exposure > 0).sum() / n_catalog),
    "n_distinct_items_recommended": int((full_exposure > 0).sum()),
    "head_exposure_share": float(exposure[is_head_by_rank[:D]].sum() / total_slots),
    "tail_exposure_share": float(1 - exposure[is_head_by_rank[:D]].sum() / total_slots),
}
for k_, v in metrics.items():
    print(f"{k_:32s} {v}")

# ---------- 6. Save ----------
with open(PROC / f"{CATEGORY}_mostpopular_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
np.save(PROC / f"{CATEGORY}_mostpopular_recs_rank.npy", recs_all)   # item ranks; map back via the ranking file
pop[["parent_asin", "cnt"]].to_parquet(PROC / f"{CATEGORY}_popularity_ranking.parquet")
print("saved")