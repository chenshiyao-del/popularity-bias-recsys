"""Matrix factorization (MF) trained with BPR loss.

Every user and item gets a learned vector of DIM numbers, and score(user, item) is their dot product.
BPR loss: for each (user, item they had, random item), push the real item's score above the random one.

Settings: DIM = vector size, EPOCHS = passes over the data, LR = learning rate, REG = L2 penalty,
TRAIN_BATCH = interactions per training step. GPU memory is roughly n_users * DIM * 12 bytes.
"""
import torch

DIM, EPOCHS, LR, REG = 16, 5, 0.005, 1e-4
TRAIN_BATCH = 8192


def build(u, i, cnt, n_users, n_items, dev, row_slicer):
    """Train the MF model.

    Args:
        u, i: user and item index of every training interaction.
        cnt: number of training interactions per item (unused here).
        n_users, n_items: number of users and items.
        dev: torch device ("cuda" or "cpu").
        row_slicer: helper from run.py for pulling sparse rows onto the GPU (unused here).

    Returns:
        score(a, b, rows): scores of every item for users a..b-1.
    """
    # Set up user/item vectors and the optimizer
    U = torch.nn.Embedding(n_users, DIM, sparse=True).to(dev)
    V = torch.nn.Embedding(n_items, DIM, sparse=True).to(dev)
    torch.nn.init.normal_(U.weight, std=0.01)
    torch.nn.init.normal_(V.weight, std=0.01)
    opt = torch.optim.SparseAdam([U.weight, V.weight], lr=LR)
    tu, ti = torch.tensor(u, device=dev), torch.tensor(i, device=dev)

    # Train with BPR loss on (user, item, random item) triples
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(tu), device=dev)
        total, steps = 0, 0
        for s in range(0, len(perm), TRAIN_BATCH):
            idx = perm[s:s + TRAIN_BATCH]
            pu, pi = U(tu[idx]), V(ti[idx])
            ni = V(torch.randint(n_items, (len(idx),), device=dev))
            loss = -torch.nn.functional.logsigmoid((pu * pi).sum(1) - (pu * ni).sum(1)).mean()
            loss = loss + REG * (pu.pow(2) + pi.pow(2) + ni.pow(2)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
            if steps % 500 == 0:
                print(f"epoch {epoch + 1}/{EPOCHS}: {s:,}/{len(perm):,} rows", end="\r")
        print(f"\nepoch {epoch + 1}/{EPOCHS} loss {total / steps:.4f}")

    # Score = user vector . item vector
    Uw, Vw = U.weight.detach(), V.weight.detach()
    return lambda a, b, rows: Uw[a:b] @ Vw.T
