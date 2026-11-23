"""Most popular: recommend the items with the most training interactions to every user."""
import torch


def build(u, i, cnt, n_users, n_items, dev, row_slicer):
    pop = torch.tensor(cnt, dtype=torch.float32, device=dev)
    return lambda a, b, rows: pop.repeat(b - a, 1)
