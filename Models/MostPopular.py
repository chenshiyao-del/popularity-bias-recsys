"""
Use by change path and category in config.py 
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DATA_DIR, CATEGORY, K, D, CHUNK, MIN_RATING, PROC
from leakage_guard import guard_all
from rec_eval import evaluate, save_metrics 

# ---------- 1. Load data + run all leakage checks ----------
# train/test are tagged by load_split(); stats is built from train only.
# Change category and path in config.py 
train, test, stats = guard_all(DATA_DIR, CATEGORY)

test = test[test["rating"] >= MIN_RATING]
#after filtering, tag "split"
test.attrs["split"] = "test" 
print(f"catalog items: {stats.n_catalog:,} | test users: {len(test):,}") 

# ---------- 2. "Fit": popularity ranking comes from train-only statistics ----------
pop = stats.items                                   # columns: parent_asin, cnt, mean_rating; row index = rank
rank_of = stats.rank_of                             # item -> popularity rank (0 = most popular)
cnt_by_rank = pop["cnt"].to_numpy()
is_head_by_rank = stats.is_head_by_rank
n_catalog = stats.n_catalog

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
recs_all = np.empty((n, K), dtype=np.int32)

for a in range(0, n, CHUNK):
    b = min(a + CHUNK, n)
    
    # build seen Matrix 
    seen = np.zeros((b - a, D), dtype=bool)
    lo, hi = np.searchsorted(su, [a, b])
    seen[su[lo:hi] - a, sr[lo:hi]] = True

    unseen = ~seen
    
    # Extract the Rank of the first K uninteracted items row by row
    for i in range(b - a):
        # np.where returns a tuple and takes the first element, which is the index array 
        valid_ranks = np.where(unseen[i])[0]
        
        # Assertion: Ensure that there are at least K unseen items within this window
        assert len(valid_ranks) >= K, f"User at index {a+i} has < {K} unseen items in top-D; increase D."
        
        # take top k items 
        recs_all[a + i] = valid_ranks[:K]
        
    print(f"  generated recommendations for {b:,}/{n:,} users", end="\r")

print("\n[MostPopular] Recommendations generated. Calling shared evaluator...")
# ---------- 5. Evaluate ----------
metrics = evaluate(
    recs=recs_all, 
    test=test, 
    train=train, 
    stats=stats, 
    K=K, 
    model_name="MostPopular"
)

for k_, v in metrics.items():
    print(f"{k_:32s} {v}")

# ---------- 6. Save ----------
save_metrics(metrics, PROC, CATEGORY)
np.save(PROC / f"{CATEGORY}_mostpopular_recs_rank.npy", recs_all)
print(f"Saved metrics and recs to {PROC}") 