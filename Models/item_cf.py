"""Item-based collaborative filtering (simple version).

Works on any REVIEW parquet (raw or cleaned) that has user_id, parent_asin, timestamp.
Meta files can't be used for training (no user_id), but can be passed with --meta
to print readable titles for a few example recommendations.

Usage:
    python item_cf.py path/to/reviews.parquet
    python item_cf.py path/to/reviews.parquet --meta path/to/meta.parquet --k 10 --sample-users 20000
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("reviews", help="review/interactions parquet file")
parser.add_argument("--meta", help="optional meta parquet, only used to show item titles")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--min-user", type=int, default=1)
parser.add_argument("--min-item", type=int, default=1)
parser.add_argument("--sample-users", type=int, default=20000, help="0 = use all users")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--neighbors", type=int, default=100,
                     help="similar items kept per item (bounds memory; raise for better recall)")
parser.add_argument("--chunk-size", type=int, default=1000,
                     help="items processed per similarity chunk (lower this if you run out of RAM)")
args = parser.parse_args()
K = args.k

# ---------- 1. load (only the needed columns) + filter ----------
print("[1/7] loading + filtering interactions...")
available = set(pq.ParquetFile(args.reviews).schema.names)
need = {"user_id", "parent_asin", "timestamp"}
if not need <= available:
    raise SystemExit(f"This file isn't a review file. Missing columns: {need - available}\n"
                     f"(Item-CF needs user_id + parent_asin + timestamp; meta files have no user_id.)")

cols = list(need) + (["verified_purchase"] if "verified_purchase" in available else [])
# push the verified_purchase filter down into the parquet read itself (pyarrow
# predicate pushdown), and read user_id/parent_asin as dictionary-encoded
# columns (-> pandas category dtype) straight away. On a 60M+ row file,
# materializing those two ID columns as plain Python strings first (even
# briefly) needs many GB just for per-object overhead; reading them pre-
# encoded avoids that entirely instead of converting to category afterward.
filters = [("verified_purchase", "==", True)] if "verified_purchase" in available else None
dict_cols = [c for c in ("user_id", "parent_asin") if c in cols]
df = pd.read_parquet(args.reviews, columns=cols, filters=filters, read_dictionary=dict_cols)
if "verified_purchase" in df:
    df = df.drop(columns="verified_purchase")
df = df.dropna()
df = df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")

for _ in range(2):                                  # a couple of passes so the filters settle
    df = df[df.groupby("user_id", observed=True)["user_id"].transform("size") >= args.min_user]
    df = df[df.groupby("parent_asin", observed=True)["parent_asin"].transform("size") >= args.min_item]

if args.sample_users:
    keep = df["user_id"].drop_duplicates()
    keep = keep.sample(min(args.sample_users, len(keep)), random_state=args.seed)
    df = df[df["user_id"].isin(keep)]

# drop now-unused categories so later .astype("category") calls reflect only
# what's actually left after filtering/sampling, not the original full set
df["user_id"] = df["user_id"].cat.remove_unused_categories()
df["parent_asin"] = df["parent_asin"].cat.remove_unused_categories()
print(f"{len(df):,} interactions, {df.user_id.nunique():,} users, {df.parent_asin.nunique():,} items")

# ---------- 2. split: each user's most recent item is the test item ----------
print("[2/7] splitting train/test...")
test = df.groupby("user_id", observed=True).tail(1)
train = df.drop(test.index)

# ---------- 3. user x item matrix (1 = interacted) ----------
print("[3/7] building user x item matrix...")
user_ids = train["user_id"].astype("category")
item_ids = train["parent_asin"].astype("category")
n_users, n_items = len(user_ids.cat.categories), len(item_ids.cat.categories)
R = csr_matrix((np.ones(len(train), dtype="float32"),
                (user_ids.cat.codes, item_ids.cat.codes)), shape=(n_users, n_items))
item_index = {a: i for i, a in enumerate(item_ids.cat.categories)}
user_index = {u: i for i, u in enumerate(user_ids.cat.categories)}
print(f"      matrix: {n_users:,} users x {n_items:,} items")

# ---------- 4. item-item similarity (cosine between item columns) ----------
# Computed in row chunks and immediately truncated to each item's top-N
# neighbors, so memory stays O(n_items * neighbors) instead of O(n_items^2)
# (a full dense/near-dense matrix), which is what makes this scale to the
# whole dataset rather than just a user sample.
print(f"[4/7] computing item-item similarity ({n_items:,} items, "
      f"chunk size {args.chunk_size}, keeping top {args.neighbors} neighbors/item)...")
R_items = normalize(R.T.tocsr(), axis=1)  # L2-normalized item vectors -> dot product = cosine
n_keep = min(args.neighbors, n_items - 1) if n_items > 1 else 0

n_chunks = (n_items + args.chunk_size - 1) // max(args.chunk_size, 1)
rows_acc, cols_acc, vals_acc = [], [], []
for chunk_i, start in enumerate(range(0, n_items, args.chunk_size), start=1):
    end = min(start + args.chunk_size, n_items)
    block = (R_items[start:end] @ R_items.T).toarray()
    np.fill_diagonal(block[:, start:end], 0)  # zero self-similarity for this chunk
    if n_keep:
        top = np.argpartition(-block, n_keep, axis=1)[:, :n_keep]
        top_vals = np.take_along_axis(block, top, axis=1)
        rows_acc.append(np.repeat(np.arange(start, end), n_keep))
        cols_acc.append(top.ravel())
        vals_acc.append(top_vals.ravel())
    if chunk_i == 1 or chunk_i == n_chunks or chunk_i % 10 == 0:
        print(f"      chunk {chunk_i}/{n_chunks} ({end:,}/{n_items:,} items done)")

print("      assembling sparse similarity matrix...")
sim = csr_matrix(
    (np.concatenate(vals_acc), (np.concatenate(rows_acc), np.concatenate(cols_acc))),
    shape=(n_items, n_items),
) if rows_acc else csr_matrix((n_items, n_items))
sim.eliminate_zeros()
print(f"      sim matrix: {sim.nnz:,} nonzero entries")

# ---------- 5. recommend: score = sum of similarity to items the user already has ----------
def recommend(user_rows, k=K):
    seen = R[user_rows]
    scores = (seen @ sim).toarray()
    scores[seen.toarray() > 0] = -np.inf            # don't re-recommend seen items
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
    return np.take_along_axis(top, order, 1)

# ---------- 6. evaluate ----------
print("[5/7] preparing test set...")
test = test[test["user_id"].isin(user_index) & test["parent_asin"].isin(item_index)]
rows = test["user_id"].map(user_index).to_numpy()
truth = test["parent_asin"].map(item_index).to_numpy()

print(f"[6/7] generating + scoring recommendations for {len(rows):,} test users...")
batches = list(range(0, len(rows), 500))
rec_batches = []
for b_i, i in enumerate(batches, start=1):
    rec_batches.append(recommend(rows[i:i + 500]))
    if b_i == 1 or b_i == len(batches) or b_i % 20 == 0:
        print(f"      batch {b_i}/{len(batches)} ({min(i + 500, len(rows)):,}/{len(rows):,} users done)")
recs = np.vstack(rec_batches)
hit_rate = (recs == truth[:, None]).any(axis=1).mean()   # one test item per user -> same as Recall@K

print("[7/7] computing metrics...")
item_pop = np.asarray(R.sum(axis=0)).ravel()
head = item_pop >= np.quantile(item_pop, 0.8)            # top 20% most popular items
print(f"\nHitRate@{K}:                {hit_rate:.4f}")
print(f"Avg popularity of recs:     {item_pop[recs].mean():.1f}  (catalog avg {item_pop.mean():.1f})")
print(f"Share of recs from top 20%: {head[recs].mean():.2%}")
print(f"Item coverage:              {len(np.unique(recs)) / n_items:.2%}")

# ---------- 7. optional: show readable examples using the meta file ----------
if args.meta:
    meta = pd.read_parquet(args.meta, columns=["parent_asin", "title"]).drop_duplicates("parent_asin")
    titles = meta.set_index("parent_asin")["title"]
    cats = item_ids.cat.categories
    print("\nExample recommendations:")
    for j in range(min(3, len(rows))):
        history = cats[R[rows[j]].indices]
        print(f"\nUser has: {[titles.get(a, a)[:40] for a in history[:3]]}")
        for idx in recs[j][:5]:
            print("  ->", titles.get(cats[idx], cats[idx])[:70])
