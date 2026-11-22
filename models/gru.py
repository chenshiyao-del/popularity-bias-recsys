"""GRU sequence recommender (GRU4Rec-style).

A GRU reads a user's most recent items in time order, and its final output is the user's vector.
score(user, item) = user vector . item vector. It is trained to predict each next item in the
history with BPR loss against a random item. It relies on train rows being in time order within
each user, which is how split.py writes them. See models/GRU_README.md for details.

Settings: DIM = vector size, EPOCHS = passes over the users, LR = learning rate,
MAX_LEN = recent items read per user, TRAIN_BATCH = users per training step.
"""
import numpy as np
import pandas as pd
import torch

DIM, EPOCHS, LR = 64, 5, 0.001
MAX_LEN = 10
TRAIN_BATCH = 1024


def build(u, i, cnt, n_users, n_items, dev, row_slicer):
    """Train the GRU model.

    Args:
        u, i: user and item index of every training interaction, in time order within each user.
        cnt: number of training interactions per item (unused here).
        n_users, n_items: number of users and items.
        dev: torch device ("cuda" or "cpu").
        row_slicer: helper from run.py for pulling sparse rows onto the GPU (unused here).

    Returns:
        score(a, b, rows): scores of every item for users a..b-1.
    """
    # Build each user's history: last MAX_LEN+1 items, newest on the right, padded in front
    PAD = n_items
    back = pd.Series(u).groupby(u).cumcount(ascending=False).to_numpy()
    keep = back <= MAX_LEN
    seq = np.full((n_users, MAX_LEN + 1), PAD)
    seq[u[keep], MAX_LEN - back[keep]] = i[keep]
    seq = torch.tensor(seq, device=dev)

    # Set up item vectors, the GRU and the optimizer
    V = torch.nn.Embedding(n_items + 1, DIM, padding_idx=PAD).to(dev)
    gru = torch.nn.GRU(DIM, DIM, batch_first=True).to(dev)
    opt = torch.optim.Adam(list(V.parameters()) + list(gru.parameters()), lr=LR)
    learners = torch.nonzero(seq[:, -2] != PAD).squeeze(1)

    # Train: predict each next item from the items before it (BPR loss vs a random item)
    for epoch in range(EPOCHS):
        learners = learners[torch.randperm(len(learners), device=dev)]
        total, steps = 0, 0
        for s in range(0, len(learners), TRAIN_BATCH):
            x = seq[learners[s:s + TRAIN_BATCH]]
            inp, tgt = x[:, :-1], x[:, 1:]
            out, _ = gru(V(inp))
            neg = torch.randint(n_items, tgt.shape, device=dev)
            diff = (out * V(tgt)).sum(-1) - (out * V(neg)).sum(-1)
            loss = -torch.nn.functional.logsigmoid(diff)[inp != PAD].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
            if steps % 500 == 0:
                print(f"epoch {epoch + 1}/{EPOCHS}: {s:,}/{len(learners):,} users", end="\r")
        print(f"\nepoch {epoch + 1}/{EPOCHS} loss {total / steps:.4f}")

    # Score = GRU output after reading the user's recent items . item vector
    def score(a, b, rows):
        with torch.no_grad():
            out, _ = gru(V(seq[a:b, 1:]))
            return out[:, -1] @ V.weight[:n_items].T
    return score
