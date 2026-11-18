"""
leakage_guard.py -- shared leakage protection for the popularity-bias project.

Rules enforced here
  R1. Train/test files may contain ONLY whitelisted columns. Item-level statistics that Amazon computed
      on the full review history (average_rating, rating_number) are forbidden, because they include
      the test-period interactions.
  R2. The split itself is valid: no unseen users/items in test, no (user, item) pair in both splits,
      no duplicated pairs in train, and every test interaction is not earlier than that user's train history.
  R3. Every statistic (popularity, ranking, head/tail, item mean rating, user profile) is computed from the
      TRAIN split only. Functions refuse any DataFrame that was not loaded through load_split(..., "train").
  R4. The head-item file on disk must match the head set recomputed from train (guards against stale files).
  R5. Item metadata joined into any model must be static features only.

Usage
    from leakage_guard import guard_all
    train, test, stats = guard_all(PROC, CATEGORY)
"""
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    pass


# ---------- Configuration ----------
ALLOWED_SPLIT_COLS = {"user_id", "parent_asin", "rating", "timestamp"}
# Computed by Amazon over the FULL history (test period included) -> never use as feature, weight or popularity.
LEAKY_COLS = {"average_rating", "rating_number"}
# Static item attributes that are safe to join (price is a scrape-time snapshot; mention it in the report).
STATIC_META_COLS = ["parent_asin", "main_category", "store", "price", "details", "categories"]


# ---------- R1: file-level column whitelist ----------
def _split_path(proc, category, split):
    for ext in ("parquet", "csv"):
        p = Path(proc) / f"{category}_{split}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No {split} file for {category} in {proc}")


def _file_columns(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0).columns)


def check_columns(proc, category):
    """R1: fail if a split file has leaky or unexpected columns."""
    for split in ("train", "test"):
        cols = set(_file_columns(_split_path(proc, category, split)))
        leaky = cols & LEAKY_COLS
        if leaky:
            raise LeakageError(f"{split} file contains leaky columns: {sorted(leaky)}")
        extra = cols - ALLOWED_SPLIT_COLS
        if extra:
            raise LeakageError(f"{split} file has unreviewed columns {sorted(extra)}; "
                               f"review them for leakage, then add to ALLOWED_SPLIT_COLS.")
        missing = ALLOWED_SPLIT_COLS - cols
        if missing:
            raise LeakageError(f"{split} file is missing columns: {sorted(missing)}")


def load_split(proc, category, split):
    """Load train or test and tag it, so downstream statistics can verify where the data came from."""
    check_columns(proc, category)
    p = _split_path(proc, category, split)
    cols = sorted(ALLOWED_SPLIT_COLS)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=cols)
    else:
        df = pd.read_csv(p, usecols=cols, parse_dates=["timestamp"])
    df = df.reset_index(drop=True)
    df.attrs["split"] = split
    return df


# ---------- R3: only train may feed statistics ----------
def require_split(df, expected):
    got = df.attrs.get("split") if hasattr(df, "attrs") else None
    if got != expected:
        raise LeakageError(f"Expected the '{expected}' split but got '{got}'. "
                           f"Load data with load_split(); do not concat/merge train with test.")


# ---------- R2: split integrity ----------
def check_split(train, test):
    require_split(train, "train")
    require_split(test, "test")

    users = pd.Index(train["user_id"].unique())
    items = pd.Index(train["parent_asin"].unique())
    tr_u = users.get_indexer(train["user_id"])
    tr_i = items.get_indexer(train["parent_asin"])
    te_u = users.get_indexer(test["user_id"])
    te_i = items.get_indexer(test["parent_asin"])

    n_cold_users, n_cold_items = int((te_u < 0).sum()), int((te_i < 0).sum())
    if n_cold_users or n_cold_items:
        raise LeakageError(f"Test has {n_cold_users} rows with users and {n_cold_items} rows with items unseen in train.")

    n_items = len(items)
    tr_key = tr_u.astype(np.int64) * n_items + tr_i
    te_key = te_u.astype(np.int64) * n_items + te_i
    dup_train = int(len(tr_key) - len(np.unique(tr_key)))
    overlap = int(np.isin(te_key, tr_key).sum())
    dup_test_users = int(test["user_id"].duplicated().sum())

    last_train = pd.Series(train["timestamp"].to_numpy()).groupby(tr_u).max()
    last = last_train.reindex(te_u).to_numpy()
    violations = int((test["timestamp"].to_numpy() < last).sum())

    problems = {"duplicated (user,item) in train": dup_train,
                "test (user,item) also in train": overlap,
                "users with >1 test row": dup_test_users,
                "test earlier than user's train history": violations}
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        raise LeakageError(f"Split integrity failed: {bad}")
    print(f"[guard] split OK | train {len(train):,} rows | test {len(test):,} rows")


# ---------- R3: statistics computed from train only ----------
class TrainOnlyStats:
    """Every popularity-derived or rating-derived feature lives here and is built from train only."""

    def __init__(self, train, head_share=0.20):
        require_split(train, "train")
        items = (train.groupby("parent_asin")["rating"].agg(cnt="size", mean_rating="mean").reset_index()
                      .sort_values(["cnt", "parent_asin"], ascending=[False, True], ignore_index=True))
        self.items = items                                   # row index == popularity rank (0 = most popular)
        self.rank_of = pd.Series(items.index.to_numpy(), index=items["parent_asin"])
        self.cnt_of = pd.Series(items["cnt"].to_numpy(), index=items["parent_asin"])
        self.item_mean_rating = pd.Series(items["mean_rating"].to_numpy(), index=items["parent_asin"])
        cum = items["cnt"].cumsum() / items["cnt"].sum()
        self.head_items = set(items.loc[cum <= head_share, "parent_asin"])
        self.is_head_by_rank = items["parent_asin"].isin(self.head_items).to_numpy()
        self.n_catalog = len(items)

    def user_profile(self, train):
        """Per-user average item popularity and tail ratio, from train interactions only."""
        require_split(train, "train")
        tmp = pd.DataFrame({
            "user_id": train["user_id"].to_numpy(),
            "pop": train["parent_asin"].map(self.cnt_of).to_numpy(),
            "is_tail": ~train["parent_asin"].isin(self.head_items).to_numpy(),
        })
        return tmp.groupby("user_id").agg(avg_item_pop=("pop", "mean"), tail_ratio=("is_tail", "mean"))


# ---------- R4: stale head file ----------
def check_head_items(proc, category, stats, tol=0.005):
    """The head file on disk must match the head set recomputed from train."""
    p = _split_path(proc, category, "head_items")
    disk = set(pd.read_parquet(p)["parent_asin"] if p.suffix == ".parquet" else pd.read_csv(p)["parent_asin"])
    diff = len(disk ^ stats.head_items)
    if diff > tol * max(len(stats.head_items), 1):
        raise LeakageError(f"head_items file ({len(disk):,}) does not match the train-based head set "
                           f"({len(stats.head_items):,}); {diff:,} items differ. The file may be stale "
                           f"(e.g. computed on the full data). Regenerate it from train.")
    if diff:
        print(f"[guard] head_items: {diff} boundary item(s) differ from the recomputed set (within tolerance).")
    else:
        print(f"[guard] head_items OK ({len(disk):,} items)")


# ---------- R5: metadata ----------
def safe_meta(meta):
    """Return only static item attributes; drop full-history statistics."""
    leaky = sorted(set(meta.columns) & LEAKY_COLS)
    if leaky:
        print(f"[guard] dropped leaky meta columns: {leaky}")
    return meta[[c for c in STATIC_META_COLS if c in meta.columns]].copy()


# ---------- One call to run everything ----------
def guard_all(proc, category):
    check_columns(proc, category)
    train = load_split(proc, category, "train")
    test = load_split(proc, category, "test")
    check_split(train, test)
    stats = TrainOnlyStats(train)
    check_head_items(proc, category, stats)
    print("[guard] all checks passed")
    return train, test, stats