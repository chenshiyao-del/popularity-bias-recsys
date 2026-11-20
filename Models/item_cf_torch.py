"""Item-based collaborative filtering, GPU version (PyTorch).

Same pipeline, split and metrics as item_cf.py, but the two heavy steps
(item-item similarity and scoring users) run on the GPU with torch sparse ops.
Falls back to CPU if CUDA isn't available.

Usage:
    python item_cf_torch.py path/to/reviews.parquet
    python item_cf_torch.py path/to/reviews.parquet --meta path/to/meta.parquet --k 10 --sample-users 0
"""
import argparse
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("reviews", help="review/interactions parquet file")
parser.add_argument("--meta", help="optional meta parquet, only used to show item titles")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--sample-users", type=int, default=20000, help="0 = use all users")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--neighbors", type=int, default=100,
                    help="similar items kept per item (bounds memory; raise for better recall)")
parser.add_argument("--chunk-size", type=int, default=0,
                    help="items per similarity chunk / users per scoring batch; 0 = auto from --gpu-mem-gb")
parser.add_argument("--gpu-mem-gb", type=float, default=1.0,
                    help="rough GPU memory budget for one dense chunk (auto chunk sizing)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
K = args.k
device = torch.device(args.device)
torch.sparse.check_sparse_tensor_invariants.disable()   # explicit opt-out; silences a startup warning
print(f"using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else
                                   "  (CUDA not available -> running on CPU)"))


def to_torch(m):
    """scipy sparse -> torch sparse COO tensor on `device`."""
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    val = torch.from_numpy(m.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, m.shape, device=device).coalesce()


# ---------- 1. load (only the needed columns) + filter ----------
print("[1/7] loading + filtering interactions...")
available = set(pq.ParquetFile(args.reviews).schema.names)
need = {"user_id", "parent_asin", "timestamp"}
if not need <= available:
    raise SystemExit(f"This file isn't a review file. Missing columns: {need - available}\n"
                     f"(Item-CF needs user_id + parent_asin + timestamp; meta files have no user_id.)")

cols = list(need)   # no cleaning here: the input is expected to be already cleaned

# Stream the file one row group at a time. Reading a 60M+ row file in one go
# (pd.read_parquet) asks for multi-GB contiguous buffers and dies with
# ArrowMemoryError. When --sample-users is set, users are chosen by hashing
# user_id, so a user is kept or dropped consistently across ALL row groups and
# only the sampled rows are ever held in memory.
pf = pq.ParquetFile(args.reviews)
n_rows = pf.metadata.num_rows
frac = 1.0
if args.sample_users:
    first = pf.read_row_group(0, columns=["user_id"]).to_pandas()["user_id"]
    est_users = max(1, int(first.nunique() / max(len(first), 1) * n_rows))   # rough distinct-user estimate
    frac = min(1.0, 1.3 * args.sample_users / est_users)                     # oversample, trimmed below
    print(f"      ~{est_users:,} users estimated -> keeping a {frac:.4%} hash slice, then trimming to {args.sample_users:,}")
elif n_rows > 20_000_000:
    print("      WARNING: --sample-users 0 on a very large file may still run out of RAM.")
HASH_LIMIT = int(frac * 2**64) if frac < 1.0 else None

parts = []
for g in range(pf.num_row_groups):
    chunk_df = pf.read_row_group(g, columns=cols).to_pandas()
    if HASH_LIMIT is not None:
        h = pd.util.hash_pandas_object(chunk_df["user_id"], index=False).to_numpy()
        chunk_df = chunk_df[h < np.uint64(HASH_LIMIT)]
    parts.append(chunk_df)
    if (g + 1) % 10 == 0 or g + 1 == pf.num_row_groups:
        print(f"      row group {g + 1}/{pf.num_row_groups}, kept {sum(len(p) for p in parts):,} rows so far", flush=True)
df = pd.concat(parts, ignore_index=True)
del parts
df["user_id"] = df["user_id"].astype(str).astype("category")
df["parent_asin"] = df["parent_asin"].astype(str).astype("category")
df = df.sort_values("timestamp")                    # only ordering, needed for the leave-last-out split

if args.sample_users:
    keep = df["user_id"].drop_duplicates()
    keep = keep.sample(min(args.sample_users, len(keep)), random_state=args.seed)
    df = df[df["user_id"].isin(keep)]

df["user_id"] = df["user_id"].cat.remove_unused_categories()
df["parent_asin"] = df["parent_asin"].cat.remove_unused_categories()
print(f"{len(df):,} interactions, {df.user_id.nunique():,} users, {df.parent_asin.nunique():,} items")

# ---------- 2. split: each user's most recent item is the test item ----------
print("[2/7] splitting train/test...")
test = df.groupby("user_id", observed=True).tail(1)
train = df.drop(test.index)

# ---------- 3. user x item matrix (1 = interacted) ----------
print("[3/7] building user x item matrix...")
user_ids = train["user_id"].astype("category").cat.remove_unused_categories()
item_ids = train["parent_asin"].astype("category").cat.remove_unused_categories()
n_users, n_items = len(user_ids.cat.categories), len(item_ids.cat.categories)
R = csr_matrix((np.ones(len(train), dtype="float32"),
                (user_ids.cat.codes, item_ids.cat.codes)), shape=(n_users, n_items))
R.data[:] = 1.0                                     # binary interaction matrix (guards against repeated pairs)
item_index = {a: i for i, a in enumerate(item_ids.cat.categories)}
user_index = {u: i for i, u in enumerate(user_ids.cat.categories)}
print(f"      matrix: {n_users:,} users x {n_items:,} items")

# chunk size: a dense (chunk x n_items) float32 block, with ~3x headroom for topk etc.
chunk = args.chunk_size or max(1, int(args.gpu_mem_gb * 1024**3 / (4 * n_items * 3)))

# ---------- 4. item-item similarity (cosine between item columns) on GPU ----------
# Item vectors are L2-normalized, so a dot product = cosine. Each chunk of rows
# is a sparse x sparse matmul on the GPU, densified, and cut to its top-N
# neighbors right away, so memory stays O(n_items * neighbors).
t0 = time.time()
R_items = normalize(R.T.tocsr(), axis=1)            # items x users (CPU, cheap row slicing)
R_users_gpu = to_torch(R_items.T)                   # users x items, lives on the GPU
n_keep = min(args.neighbors, n_items - 1) if n_items > 1 else 0
n_chunks = (n_items + chunk - 1) // chunk
print(f"[4/7] computing item-item similarity ({n_items:,} items, "
      f"chunk size {chunk}, keeping top {n_keep} neighbors/item)...")

rows_acc, cols_acc, vals_acc = [], [], []
with torch.no_grad():
    for chunk_i, start in enumerate(range(0, n_items, chunk), start=1):
        end = min(start + chunk, n_items)
        block = torch.sparse.mm(to_torch(R_items[start:end]), R_users_gpu).to_dense()
        local = torch.arange(end - start, device=device)
        block[local, local + start] = -1.0          # never pick an item as its own neighbor
        if n_keep:
            vals, top = torch.topk(block, n_keep, dim=1)
            keep_mask = vals > 0                    # drop zero / self entries
            rows_acc.append((local + start).unsqueeze(1).expand_as(top)[keep_mask].cpu())
            cols_acc.append(top[keep_mask].cpu())
            vals_acc.append(vals[keep_mask].cpu())
        del block
        if chunk_i == 1 or chunk_i == n_chunks or chunk_i % 10 == 0:
            print(f"      chunk {chunk_i}/{n_chunks} ({end:,}/{n_items:,} items done)")

    print("      assembling sparse similarity matrix...")
    if rows_acc:
        idx = torch.stack([torch.cat(rows_acc), torch.cat(cols_acc)])
        sim = torch.sparse_coo_tensor(idx, torch.cat(vals_acc), (n_items, n_items)).coalesce().to(device)
    else:
        sim = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0),
                                      (n_items, n_items)).to(device)
    del R_users_gpu, rows_acc, cols_acc, vals_acc
print(f"      sim matrix: {sim._nnz():,} nonzero entries  ({time.time() - t0:.1f}s)")


# ---------- 5. recommend: score = sum of similarity to items the user already has ----------
def recommend(user_rows, k=K):
    seen = R[user_rows]                                         # CPU scipy slice
    with torch.no_grad():
        scores = torch.sparse.mm(to_torch(seen), sim).to_dense()
        s = seen.tocoo()
        scores[torch.from_numpy(s.row).long().to(device),
               torch.from_numpy(s.col).long().to(device)] = -float("inf")   # don't re-recommend seen
        return torch.topk(scores, min(k, n_items), dim=1).indices.cpu().numpy()   # already sorted


# ---------- 6. evaluate ----------
print("[5/7] preparing test set...")
test = test[test["user_id"].isin(user_index) & test["parent_asin"].isin(item_index)]
rows = test["user_id"].map(user_index).to_numpy()
truth = test["parent_asin"].map(item_index).to_numpy()

t0 = time.time()
print(f"[6/7] generating + scoring recommendations for {len(rows):,} test users (batch {chunk})...")
batches = list(range(0, len(rows), chunk))
rec_batches = []
for b_i, i in enumerate(batches, start=1):
    rec_batches.append(recommend(rows[i:i + chunk]))
    if b_i == 1 or b_i == len(batches) or b_i % 20 == 0:
        print(f"      batch {b_i}/{len(batches)} ({min(i + chunk, len(rows)):,}/{len(rows):,} users done)")
recs = np.vstack(rec_batches) if rec_batches else np.empty((0, K), dtype=int)
print(f"      done ({time.time() - t0:.1f}s)")
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
        print(f"\nUser has: {[str(titles.get(a, a))[:40] for a in history[:3]]}")
        for idx in recs[j][:5]:
            print("  ->", str(titles.get(cats[idx], cats[idx]))[:70])
