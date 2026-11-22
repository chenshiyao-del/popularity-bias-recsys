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
parser.add_argument("--sample-users", type=int, default=0, help="0 = use all users")
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


# ---------- 1. load (only the needed columns), as compact integer arrays ----------
# The 60M+ row file is streamed one row group at a time. The two ID columns are
# hashed to uint64 right away, so the big string columns never exist in memory
# for the whole file (pandas string categories on 61M rows run out of RAM).
# Whole file as arrays: 2 x uint64 + 1 x int64 per row, ~1.5 GB at 61M rows.
# 64-bit hashes make a collision between two different IDs practically impossible
# (~1e-5 probability across 22M users, far less for items).
print("[1/7] loading interactions...")
pf = pq.ParquetFile(args.reviews)
available = set(pf.schema.names)
need = {"user_id", "parent_asin", "timestamp"}
if not need <= available:
    raise SystemExit(f"This file isn't a review file. Missing columns: {need - available}\n"
                     f"(Item-CF needs user_id + parent_asin + timestamp; meta files have no user_id.)")


def hash_ids(series):
    return pd.util.hash_pandas_object(series, index=False).to_numpy()


n_rows = pf.metadata.num_rows
u_hash = np.empty(n_rows, dtype=np.uint64)
i_hash = np.empty(n_rows, dtype=np.uint64)
ts = np.empty(n_rows, dtype=np.int64)
pos = 0
for g in range(pf.num_row_groups):
    chunk_df = pf.read_row_group(g, columns=sorted(need)).to_pandas()
    m = len(chunk_df)
    u_hash[pos:pos + m] = hash_ids(chunk_df["user_id"])
    i_hash[pos:pos + m] = hash_ids(chunk_df["parent_asin"])
    ts[pos:pos + m] = chunk_df["timestamp"].astype("int64").to_numpy()
    pos += m
    del chunk_df
    if (g + 1) % 10 == 0 or g + 1 == pf.num_row_groups:
        print(f"      row group {g + 1}/{pf.num_row_groups}, loaded {pos:,} rows", flush=True)
u_hash, i_hash, ts = u_hash[:pos], i_hash[:pos], ts[:pos]

u_all, u_uniq = pd.factorize(u_hash)                # int codes 0..n_users_all-1
del u_hash
n_users_all = len(u_uniq)

if args.sample_users and args.sample_users < n_users_all:      # exact number of users, chosen by seed
    rng = np.random.default_rng(args.seed)
    keep_u = np.zeros(n_users_all, dtype=bool)
    keep_u[rng.choice(n_users_all, args.sample_users, replace=False)] = True
    m = keep_u[u_all]
    u_all, i_hash, ts = u_all[m], i_hash[m], ts[m]
    del keep_u, m

i_all, i_uniq = pd.factorize(i_hash)                # int codes 0..n_items_all-1 (i_uniq = item hashes)
del i_hash

order = np.argsort(ts, kind="stable")               # only ordering, needed for the leave-last-out split
u_all = u_all[order].astype(np.int32)
i_all = i_all[order].astype(np.int32)
del ts, order
print(f"{len(u_all):,} interactions, {len(np.unique(u_all)):,} users, {len(np.unique(i_all)):,} items")

# ---------- 2. split: each user's most recent item is the test item ----------
print("[2/7] splitting train/test...")
last = np.full(n_users_all, -1, dtype=np.int64)
last[u_all] = np.arange(len(u_all))                 # repeated index: the last write wins = most recent row
is_test = np.zeros(len(u_all), dtype=bool)
is_test[last[last >= 0]] = True
test_u, test_i = u_all[is_test], i_all[is_test]
tr_u, tr_i = u_all[~is_test], i_all[~is_test]
del u_all, i_all, is_test, last

# ---------- 3. user x item matrix (1 = interacted) ----------
print("[3/7] building user x item matrix...")
tr_u_idx, tr_u_orig = pd.factorize(tr_u)            # only users / items that appear in train get a row / column
tr_i_idx, tr_i_orig = pd.factorize(tr_i)
n_users, n_items = len(tr_u_orig), len(tr_i_orig)
R = csr_matrix((np.ones(len(tr_u), dtype="float32"), (tr_u_idx, tr_i_idx)), shape=(n_users, n_items))
R.data[:] = 1.0                                     # binary interaction matrix (guards against repeated pairs)
item_hash = i_uniq[tr_i_orig]                       # hash of each matrix column, to look up names later
u_map = np.full(n_users_all, -1, dtype=np.int64)    # original user code -> matrix row (-1 = no train history)
u_map[tr_u_orig] = np.arange(n_users)
i_map = np.full(len(i_uniq), -1, dtype=np.int64)    # original item code -> matrix column (-1 = not in train)
i_map[tr_i_orig] = np.arange(n_items)
del tr_u, tr_i, tr_u_idx, tr_i_idx
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
rows, truth = u_map[test_u], i_map[test_i]          # users with train history and test items seen in train
ok = (rows >= 0) & (truth >= 0)
rows, truth = rows[ok], truth[ok]

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


def report_metrics(hit_rate, recs, head, item_pop, n_items, k):
    """Print the evaluation metrics and return them as a dict."""
    metrics = {
        f"hit_rate@{k}": float(hit_rate),
        "avg_rec_popularity": float(item_pop[recs].mean()),
        "catalog_avg_popularity": float(item_pop.mean()),
        "share_recs_top20pct": float(head[recs].mean()),
        "item_coverage": float(len(np.unique(recs)) / n_items),
    }
    print(f"\nHitRate@{k}:                {metrics[f'hit_rate@{k}']:.4f}")
    print(f"Avg popularity of recs:     {metrics['avg_rec_popularity']:.1f}  (catalog avg {metrics['catalog_avg_popularity']:.1f})")
    print(f"Share of recs from top 20%: {metrics['share_recs_top20pct']:.2%}")
    print(f"Item coverage:              {metrics['item_coverage']:.2%}")
    return metrics


metrics = report_metrics(hit_rate, recs, head, item_pop, n_items, K)

# ---------- 7. optional: show readable examples using the meta file ----------
if args.meta:
    meta = pd.read_parquet(args.meta, columns=["parent_asin", "title"]).drop_duplicates("parent_asin")
    titles = meta.set_index("parent_asin")["title"]
    n_show = min(3, len(rows))
    hist = [R[rows[j]].indices[:3] for j in range(n_show)]
    shown = sorted({int(c) for j in range(n_show) for c in np.concatenate([hist[j], recs[j][:5]])})

    # IDs were hashed while loading; recover the ASIN strings for just these few items
    wanted = item_hash[shown]
    asin_of = {}
    for g in range(pf.num_row_groups):
        col = pf.read_row_group(g, columns=["parent_asin"]).to_pandas()["parent_asin"]
        h = hash_ids(col)
        hit = np.isin(h, wanted)
        for a, hh in zip(col[hit], h[hit]):
            asin_of[int(hh)] = str(a)
        if len(asin_of) == len(set(wanted.tolist())):
            break
    name = lambda c: str(titles.get(asin_of.get(int(item_hash[c])), asin_of.get(int(item_hash[c]))))

    print("\nExample recommendations:")
    for j in range(n_show):
        print(f"\nUser has: {[name(c)[:40] for c in hist[j]]}")
        for idx in recs[j][:5]:
            print("  ->", name(idx)[:70])
