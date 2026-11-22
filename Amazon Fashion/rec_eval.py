"""
rec_eval.py -- shared evaluation for ALL models (MostPopular, ItemCF, WMF, NCF).

Contract every model must follow
  * Item index space = popularity rank from TrainOnlyStats (stats.rank_of): 0 = most popular training item.
    A model with its own item ids converts with:  ranks = stats.rank_of.loc[item_ids].to_numpy()
  * recs: int array of shape (n_test_users, K); row i belongs to test.iloc[i]; column 0 = best item.
  * Items the user already interacted with in TRAIN must not be recommended (checked below).

Usage
    from leakage_guard import guard_all
    from rec_eval import evaluate, sample_test_users, save_metrics
    train, test, stats = guard_all(PROC, CATEGORY)
    metrics = evaluate(recs, test, train, stats, K=10, model_name="ItemCF")
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leakage_guard import LeakageError, require_split


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n, cum = len(x), np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def sample_test_users(test, n=100_000, seed=42):
    """Fixed random subset of test users, so expensive models (e.g. NCF) are all evaluated on the SAME users."""
    require_split(test, "test")
    idx = np.sort(np.random.default_rng(seed).choice(len(test), size=min(n, len(test)), replace=False))
    sub = test.iloc[idx].reset_index(drop=True)
    sub.attrs["split"] = "test"
    return sub, idx


def _validate_recs(recs, test, train, stats, K, chunk=500_000):
    n, n_cat = len(test), stats.n_catalog
    if recs.shape != (n, K):
        raise ValueError(f"recs must have shape ({n}, {K}) = (n_test_users, K), got {recs.shape}")
    if not np.issubdtype(recs.dtype, np.integer):
        raise ValueError("recs must be an integer array of popularity ranks")
    if recs.min() < 0 or recs.max() >= n_cat:
        raise ValueError(f"recs contain ids outside [0, {n_cat}). Convert item ids with stats.rank_of.")

    # (a) no duplicated items inside one list
    for a in range(0, n, chunk):
        s = np.sort(recs[a:a + chunk], axis=1)
        if (s[:, 1:] == s[:, :-1]).any():
            raise ValueError("Some recommendation lists contain the same item more than once.")

    # (b) no already-seen training item is recommended
    user_row = pd.Series(np.arange(n), index=test["user_id"])
    tr_u = train["user_id"].map(user_row)
    m = tr_u.notna().to_numpy()
    tr_key = (tr_u[m].to_numpy().astype(np.int64) * n_cat
              + train["parent_asin"].map(stats.rank_of).to_numpy()[m].astype(np.int64))
    tr_key.sort()
    n_seen = 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        keys = ((np.arange(a, b, dtype=np.int64)[:, None] * n_cat) + recs[a:b]).ravel()
        pos = np.searchsorted(tr_key, keys)
        n_seen += int((tr_key[np.minimum(pos, len(tr_key) - 1)] == keys).sum())
    if n_seen:
        raise LeakageError(f"{n_seen:,} recommended (user, item) pairs are already in the user's training history. "
                           f"Filter seen items before recommending.")


def evaluate(recs, test, train, stats, K=10, model_name="model", return_per_user=False):
    require_split(test, "test")
    require_split(train, "train")
    recs = np.asarray(recs)
    _validate_recs(recs, test, train, stats, K)

    n, n_cat = len(test), stats.n_catalog
    t_rank = test["parent_asin"].map(stats.rank_of).to_numpy().astype(np.int64)

    # ---- accuracy (one held-out item per user => Recall@K == HitRate@K) ----
    match = recs == t_rank[:, None]
    hit = match.any(1)
    pos = match.argmax(1)
    ndcg = np.where(hit, 1.0 / np.log2(pos + 2), 0.0)
    test_is_head = stats.is_head_by_rank[t_rank]

    # ---- bias: computed on exposure over the FULL catalog ----
    exposure = np.bincount(recs.ravel(), minlength=n_cat).astype(float)
    cnt = stats.items["cnt"].to_numpy()
    head_exp = exposure[stats.is_head_by_rank].sum() / (n * K)

    metrics = {
        "model": model_name, "K": K, "n_users": int(n),
        f"Recall@{K}": float(hit.mean()),
        f"NDCG@{K}": float(ndcg.mean()),
        f"Recall@{K}_head_test_items": float(hit[test_is_head].mean()) if test_is_head.any() else float("nan"),
        f"Recall@{K}_tail_test_items": float(hit[~test_is_head].mean()) if (~test_is_head).any() else float("nan"),
        "share_of_test_items_in_head": float(test_is_head.mean()),
        "ARP": float((exposure * cnt).sum() / (n * K)),               # avg training popularity of recommended items
        "ARP_test_items": float(cnt[t_rank].mean()),                  # reference: popularity of what users really consumed
        "Gini_exposure": float(gini(exposure)),
        "Coverage": float((exposure > 0).sum() / n_cat),
        "n_distinct_items_recommended": int((exposure > 0).sum()),
        "head_exposure_share": float(head_exp),
        "tail_exposure_share": float(1 - head_exp),
    }
    if return_per_user:
        return metrics, {"hit": hit, "ndcg": ndcg}
    return metrics


def save_metrics(metrics, proc, category):
    path = Path(proc) / f"{category}_{metrics['model'].lower()}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path