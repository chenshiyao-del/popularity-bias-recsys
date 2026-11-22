"""Train one recommender, recommend top-K items to every test user, and report the results.

Models live in models/<name>.py. Each has a build(...) function that returns a score function.
Metrics: Recall@K and NDCG@K (accuracy), Recall split by head/tail items, and popularity-bias
measures. ARP is the average training popularity of the recommended items. Gini and Coverage
measure how evenly recommendations spread over the catalog. Head/tail exposure gives the share of
recommendations from the most popular items (the head, which together make up 20% of training
interactions) versus everything else. Each run appends one row to results.csv.

Settings: K = list length, BATCH = users scored per GPU batch (lower it if the GPU runs out of memory).

Usage: python run.py <category> <k> <mostpop|itemcf|mf|gru>
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from models import itemcf, wmf, mostpop

cat, k_core, model = sys.argv[1], sys.argv[2], sys.argv[3]
K = 10
BATCH = 256
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load the train/test split
train = pd.read_parquet(f"data/{cat}_k{k_core}_train.parquet")
test = pd.read_parquet(f"data/{cat}_k{k_core}_test.parquet")

# Map user and item ids to 0..n-1 and count item popularity
u, users = pd.factorize(train.user_id)
i, items = pd.factorize(train.parent_asin)
test_u, test_i = users.get_indexer(test.user_id), items.get_indexer(test.parent_asin)
n_users, n_items = len(users), len(items)
cnt = np.bincount(i, minlength=n_items)


def row_slicer(r, c, v, n_cols):
    """Prepare a sparse matrix so blocks of its rows can be pulled onto the GPU.

    Args:
        r, c, v: row indices, column indices and values of the matrix's non-zero entries.
        n_cols: number of columns.

    Returns:
        f(a, b): rows a..b-1 as a sparse tensor on the GPU.
    """
    R = scipy.sparse.csr_matrix((v, (r, c)), shape=(r.max() + 1, n_cols))
    def f(a, b):
        block = R[a:b].tocoo()
        idx = torch.tensor(np.vstack([block.row, block.col]), dtype=torch.long, device=dev)
        return torch.sparse_coo_tensor(idx, torch.tensor(block.data, device=dev), block.shape).coalesce()
    return f


user_rows = row_slicer(u, i, np.ones(len(u), dtype=np.float32), n_items)

# Build / train the chosen model
if model == "mostpop":
    score = mostpop.build(u, i, cnt, n_users, n_items, dev, row_slicer)
elif model == "itemcf":
    score = itemcf.build(u, i, cnt, n_users, n_items, dev, row_slicer)
elif model == "wmf":
    score = mf.build(u, i, cnt, n_users, n_items, dev, row_slicer, score = wmf.build(u, i, cnt, n_users, n_items, dev, row_slicer,
                      r=train["rating"].to_numpy()))


# Recommend the top-K unseen items for every user
print()
recs = np.empty((n_users, K), dtype=np.int64)
for a in range(0, n_users, BATCH):
    b = min(a + BATCH, n_users)
    rows = user_rows(a, b)
    s = score(a, b, rows)
    s[rows.indices()[0], rows.indices()[1]] = -float("inf")
    recs[a:b] = s.topk(K, dim=1).indices.cpu().numpy()
    print(f"recommending: {b:,}/{n_users:,} users", end="\r")
recs = recs[test_u]
print()

# Mark head items (most popular items making up 20% of training interactions)
order = np.argsort(-cnt, kind="stable")
head = np.zeros(n_items, dtype=bool)
head[order[np.cumsum(cnt[order]) <= 0.2 * cnt.sum()]] = True

# Compute accuracy and popularity-bias metrics
hit = recs == test_i[:, None]
found = hit.any(1)
ndcg = np.where(found, 1 / np.log2(hit.argmax(1) + 2), 0)
in_head = head[test_i]
exposure = np.sort(np.bincount(recs.ravel(), minlength=n_items))
gini = (n_items + 1 - 2 * (np.cumsum(exposure) / exposure.sum()).sum()) / n_items

metrics = {
    "category": cat, "k_core": k_core, "model": model,
    f"Recall@{K}": found.mean(),
    f"NDCG@{K}": ndcg.mean(),
    f"Recall@{K}_head_items": found[in_head].mean(),
    f"Recall@{K}_tail_items": found[~in_head].mean(),
    "test_items_in_head": in_head.mean(),
    "ARP": cnt[recs].mean(),
    "Gini_exposure": gini,
    "Coverage": (exposure > 0).mean(),
    "head_exposure": head[recs].mean(),
    "tail_exposure": 1 - head[recs].mean(),
}

# Print and save results
for key, val in metrics.items():
    print(f"{key:24} {val}")
pd.DataFrame([metrics]).to_csv("results.csv", mode="a", header=not os.path.exists("results.csv"), index=False)

# Show example recommendations with item titles (if the meta file exists)
meta_path = f"data/interim/{cat}_meta.parquet"
if os.path.exists(meta_path):
    titles = pd.read_parquet(meta_path, columns=["parent_asin", "title"]).drop_duplicates("parent_asin").set_index("parent_asin")["title"]
    name = lambda j: str(titles.get(items[j], items[j]))[:60]
    for t in range(3):
        print(f"\nuser has: {[name(j) for j in i[u == test_u[t]][:3]]}")
        for j in recs[t, :5]:
            print("  ->", name(j))
