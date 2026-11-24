"""Item-based collaborative filtering.

An item's score for a user is the sum of its cosine similarities to the items the user already has.
Only each item's top NEIGHBORS most similar items are kept.

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
    sim_r, sim_c, sim_v = [], [], []
    for a in range(0, n_items, BATCH):
        b = min(a + BATCH, n_items)
        S = torch.sparse.mm(item_rows(a, b), Rn).to_dense()
        S[torch.arange(b - a, device=dev), torch.arange(a, b, device=dev)] = 0
        v, c = S.topk(min(NEIGHBORS, n_items), dim=1)
        keep = v > 0
        sim_r.append((torch.arange(a, b, device=dev)[:, None].expand_as(c))[keep].cpu())
        sim_c.append(c[keep].cpu())
        sim_v.append(v[keep].cpu())
        print(f"similarity: {b:,}/{n_items:,} items", end="\r")
    del Rn

    # Assemble the sparse similarity matrix used for scoring
    sim = torch.sparse_coo_tensor(torch.stack([torch.cat(sim_r), torch.cat(sim_c)]), torch.cat(sim_v),
                                  (n_items, n_items)).coalesce().to(dev)
    return lambda a, b, rows: torch.sparse.mm(rows, sim).to_dense()