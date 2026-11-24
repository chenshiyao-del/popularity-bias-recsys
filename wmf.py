import time

import numpy as np
import scipy.sparse as sp
import torch

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

DIM, ITERATIONS, REG, ALPHA = 32, 10, 0.1, 20.0
MAX_BATCH_NNZ = 20_000   
SEED = 42

# rating -> (preference p, weight w)
RATING_MAP = {
    5: (1.0, 1.00),
    4: (1.0, 0.75),
    3: (1.0, 0.50),
    2: (0.0, 0.50),
    1: (0.0, 1.00),
}


# ---------------------------------------------------------------------------
# ALS solver: one half-step (solve every row of X with the other side Y fixed)
#   (Y^T Y + sum_{j in S_u} (c_uj - 1) y_j y_j^T + REG*I) x_u = sum_{j in S_u} c_uj p_uj y_j
# ---------------------------------------------------------------------------
if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _solve_rows(indptr, indices, cdata, pdata, Y, base, X):
        n, k = X.shape
        for u in prange(n):
            s, e = indptr[u], indptr[u + 1]
            if e == s:
                continue                      # no ratings -> vector stays 0
            Yo = Y[indices[s:e]]
            c = cdata[s:e]
            Yw = Yo * (c - np.float32(1.0)).reshape(-1, 1)
            A = base + np.ascontiguousarray(Yw.T) @ Yo
            b = ((c * pdata[s:e]).reshape(1, -1) @ Yo).reshape(-1)
            X[u] = np.linalg.solve(A, b)

    def _als_step(C, P, Y):
        k = Y.shape[1]
        base = (Y.T @ Y + REG * np.eye(k, dtype=np.float32)).astype(np.float32)
        X = np.zeros((C.shape[0], k), dtype=np.float32)
        _solve_rows(C.indptr, C.indices, C.data, P.data, Y, base, X)
        return X
else:
    def _als_step(C, P, Y):
        n, k = C.shape[0], Y.shape[1]
        base = Y.T @ Y + REG * np.eye(k, dtype=np.float32)
        X = np.zeros((n, k), dtype=np.float32)
        indptr = C.indptr
        start = 0
        while start < n:
            stop = int(np.searchsorted(indptr, indptr[start] + MAX_BATCH_NNZ, side="right")) - 1
            stop = min(max(stop, start + 1), n)
            lo, hi = indptr[start], indptr[stop]
            if hi == lo:
                start = stop
                continue
            has = np.diff(indptr[start:stop + 1]) > 0
            cols, c, p = C.indices[lo:hi], C.data[lo:hi], P.data[lo:hi]
            Yo = Y[cols]
            Yw = Yo * (c - 1.0)[:, None]
            if stop - start == 1:
                A = (base + Yw.T @ Yo)[None]
                b = ((c * p)[:, None] * Yo).sum(axis=0, keepdims=True)
            else:
                outer = np.einsum("ni,nj->nij", Yw, Yo)
                seg = indptr[start:stop][has] - lo
                A = base + np.add.reduceat(outer, seg, axis=0)
                b = np.add.reduceat((c * p)[:, None] * Yo, seg, axis=0)
            X[start:stop][has] = np.linalg.solve(A, b[..., None])[..., 0]
            start = stop
        return X


def _csr(rows, cols, vals, shape):
    m = sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
    m.sort_indices()
    return m


def build(u, i, cnt, n_users, n_items, dev, row_slicer, r=None):
    """Train the WMF model.

    Args:
        u, i: user and item index of every training interaction.
        cnt: number of training interactions per item (used to never recommend items with 0).
        n_users, n_items: number of users and items.
        dev: torch device ("cuda" or "cpu"); the final vectors are moved here for scoring.
        row_slicer: helper from run.py for pulling sparse rows onto the GPU (unused here).
        r: optional rating (1-5) of every training interaction, same order as u and i.
           If None, every interaction counts as a 5-star rating.

    Returns:
        score(a, b, rows): scores of every item for users a..b-1.
    """
    u = np.asarray(u, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)

    # Ratings -> preference + confidence
    if r is None:
        print("[WMF] no ratings passed -> treating every interaction as 5 stars (implicit WMF)")
        r = np.full(len(u), 5)
    r = np.clip(np.rint(np.asarray(r, dtype=float)), 1, 5).astype(int)
    pref_of = np.array([0] + [RATING_MAP[k][0] for k in range(1, 6)], dtype=np.float32)
    weight_of = np.array([0] + [RATING_MAP[k][1] for k in range(1, 6)], dtype=np.float32)
    pref = pref_of[r]
    conf = 1.0 + ALPHA * weight_of[r]

    # Sparse matrices, user-side and item-side
    C_ui = _csr(u, i, conf, (n_users, n_items))
    P_ui = _csr(u, i, pref, (n_users, n_items))
    C_iu, P_iu = C_ui.T.tocsr(), P_ui.T.tocsr()
    C_iu.sort_indices()
    P_iu.sort_indices()
    # C and P must share one sparsity pattern (P keeps explicit 0s for negatives)
    assert C_ui.nnz == P_ui.nnz == C_iu.nnz == P_iu.nnz

    # Train with ALS
    rng = np.random.default_rng(SEED)
    X = (rng.standard_normal((n_users, DIM)) * 0.01).astype(np.float32)
    Y = (rng.standard_normal((n_items, DIM)) * 0.01).astype(np.float32)
    print(f"[WMF] {len(u):,} ratings | {n_users:,} users | {n_items:,} items | "
          f"{'numba' if HAVE_NUMBA else 'numpy fallback (pip install numba for speed)'}")
    for it in range(ITERATIONS):
        t0 = time.perf_counter()
        X = _als_step(C_ui, P_ui, Y)
        Y = _als_step(C_iu, P_iu, X)
        print(f"iteration {it + 1}/{ITERATIONS} ({time.perf_counter() - t0:.1f}s)")

    # Items with no training ratings have zero vectors and would score 0, which can beat
    # real items for users whose scores are negative -> push them to the bottom.
    Uw = torch.as_tensor(X, device=dev)
    Vw = torch.as_tensor(Y, device=dev)
    no_train = None
    if cnt is not None:
        mask = np.asarray(cnt)[:n_items] == 0
        if mask.any():
            no_train = torch.as_tensor(mask, device=dev)

    # Score = user vector . item vector
    def score(a, b, rows):
        s = Uw[a:b] @ Vw.T
        if no_train is not None:
            s[:, no_train] = float("-inf")
        return s

    return score
