"""Item-based collaborative filtering.

An item's score for a user is the sum of its cosine similarities to the items the user already has.
Only each item's top NEIGHBORS most similar items are kept, stored as a CSR sparse matrix
(32-bit ids, ~8 bytes per entry) so it fits on the GPU.

Settings: NEIGHBORS = similar items kept per item, BATCH = items per GPU batch (lower it if the GPU
runs out of memory).
"""
import numpy as np
import torch

NEIGHBORS = 100
BATCH = 256


def build(u, i, cnt, n_users, n_items, dev, row_slicer):
    """Compute the item-item similarity matrix.

    Args:
        u, i: user and item index of every training interaction.
        cnt: number of training interactions per item.
        n_users, n_items: number of users and items.
        dev: torch device ("cuda" or "cpu").
        row_slicer: helper from run.py for pulling sparse rows onto the GPU.

    Returns:
        score(a, b, rows): scores of every item for users a..b-1, where rows is their interaction matrix.
    """
    # Normalize each item's column so dot products become cosine similarities
    w = (1 / np.sqrt(cnt[i])).astype(np.float32)
    Rn = torch.sparse_coo_tensor(torch.tensor(np.vstack([u, i])), torch.tensor(w), (n_users, n_items)).coalesce().to(dev)
    item_rows = row_slicer(i, u, w, n_users)

    # Compute similarities in batches of items and keep each item's top neighbors
    sim_n, sim_c, sim_v = [], [], []
    for a in range(0, n_items, BATCH):
        b = min(a + BATCH, n_items)
        S = torch.sparse.mm(item_rows(a, b), Rn).to_dense()
        S[torch.arange(b - a, device=dev), torch.arange(a, b, device=dev)] = 0
        v, c = S.topk(min(NEIGHBORS, n_items), dim=1)
        c, order = c.sort(dim=1)  # CSR needs each item's neighbors in column order
        v = v.gather(1, order)
        keep = v > 0
        sim_n.append(keep.sum(1).cpu())
        sim_c.append(c[keep].int().cpu())  # 32-bit column ids
        sim_v.append(v[keep].cpu())
        print(f"similarity: {b:,}/{n_items:,} items", end="\r")
    del Rn, S

    # Store the similarity matrix as CSR: one column id + value per entry, plus where each item's
    # row starts. With 32-bit ids that is ~8 bytes per entry (COO needs ~20), so it fits on the GPU.
    starts = torch.zeros(n_items + 1, dtype=torch.int64)
    starts[1:] = torch.cat(sim_n).cumsum(0)
    sim = torch.sparse_csr_tensor(starts.int(), torch.cat(sim_c), torch.cat(sim_v), (n_items, n_items)).to(dev)
    del sim_n, sim_c, sim_v
    print(f"\nsimilarity matrix: {sim._nnz():,} nonzero entries")
    return lambda a, b, rows: torch.sparse.mm(rows.to_sparse_csr(), sim).to_dense()
