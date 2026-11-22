"""
Shared train / val / test split. Run ONCE; every model loads the saved files.
Nobody re-splits inside their own model code.

All settings (CATEGORY, K_CORE, paths) come from config.py.

Input : data/interim/{CATEGORY}_interactions.parquet   (from the cleaning script)
Output: config.DATA_DIR = data/processed/{CATEGORY}/k{K_CORE}/
            train.parquet  val.parquet  test.parquet
            head_items.parquet   (required by leakage_guard rule R4)
            split_info.json

Order of operations:
  1. (optional) k-core filter on the FULL interactions, before splitting
  2. temporal leave-last-out per user:
       latest interaction        -> test   (users with >= 2 interactions)
       second-latest interaction -> val    (users with >= 3 interactions)
       everything earlier        -> train
  3. drop val/test rows whose item never appears in train (cold items)
  4. head items = most popular TRAIN items covering 20% of train interactions
     (same rule as leakage_guard.TrainOnlyStats)

Usage:
  python split.py            # uses K_CORE from config.py
  python split.py 5          # override, e.g. 5-core (then also update config.K_CORE)
"""
import sys
import json
import hashlib
import pandas as pd

import config

CATEGORY = config.CATEGORY
K_CORE = int(sys.argv[1]) if len(sys.argv) > 1 else config.K_CORE
HEAD_SHARE = 0.20     # must match TrainOnlyStats(head_share=...) in leakage_guard.py

IN_PATH = config.REPO_DIR / "data" / "interim" / f"{CATEGORY}_interactions.parquet"
OUT_DIR = config.PROC / CATEGORY / f"k{K_CORE}"


def describe(df, name):
    print(f"{name:>10}: {len(df):>10,} rows | {df['user_id'].nunique():>9,} users "
          f"| {df['parent_asin'].nunique():>9,} items")


def k_core(df, k):
    """Repeatedly drop users and items with < k interactions until stable.
    Repeating matters: removing a sparse item can push a user below k, and vice versa."""
    rounds = 0
    while True:
        rounds += 1
        before = len(df)
        item_n = df.groupby("parent_asin")["user_id"].transform("size")
        df = df[item_n >= k]
        user_n = df.groupby("user_id")["parent_asin"].transform("size")
        df = df[user_n >= k]
        if len(df) == before:
            print(f"k-core ({k}) converged after {rounds} rounds")
            return df


def head_items_from_train(train, head_share=HEAD_SHARE):
    """Identical rule to leakage_guard.TrainOnlyStats, so check_head_items (R4) passes."""
    items = (train.groupby("parent_asin").size().rename("cnt").reset_index()
                  .sort_values(["cnt", "parent_asin"], ascending=[False, True], ignore_index=True))
    cum = items["cnt"].cumsum() / items["cnt"].sum()
    return items.loc[cum <= head_share, ["parent_asin"]].reset_index(drop=True)


def file_hash(path):
    """Short fingerprint so teammates can confirm they have identical files."""
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. Load and (optionally) k-core filter
# ---------------------------------------------------------------------------
df = pd.read_parquet(IN_PATH, columns=["user_id", "parent_asin", "rating", "timestamp"])
describe(df, "loaded")

if K_CORE > 1:
    df = k_core(df, K_CORE)
    describe(df, f"{K_CORE}-core")

# ---------------------------------------------------------------------------
# 2. Temporal leave-last-out
#    Sort by time; parent_asin breaks timestamp ties so the split is
#    identical every time anyone runs it.
# ---------------------------------------------------------------------------
df = df.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort").reset_index(drop=True)

# position counted from the end: 0 = user's latest interaction, 1 = second-latest, ...
pos_from_end = df.groupby("user_id").cumcount(ascending=False)
n_user = df.groupby("user_id")["parent_asin"].transform("size")

is_test = (pos_from_end == 0) & (n_user >= 2)
is_val = (pos_from_end == 1) & (n_user >= 3)

test = df[is_test]
val = df[is_val]
train = df[~is_test & ~is_val]

# ---------------------------------------------------------------------------
# 3. Remove cold items from val/test: no model can recommend an item it never
#    saw in training, and leakage_guard.check_split rejects them anyway.
# ---------------------------------------------------------------------------
train_items = set(train["parent_asin"].unique())
val_cold = ~val["parent_asin"].isin(train_items)
test_cold = ~test["parent_asin"].isin(train_items)
print(f"\nCold rows dropped: val {val_cold.sum():,} ({val_cold.mean():.1%}) | "
      f"test {test_cold.sum():,} ({test_cold.mean():.1%})")
val, test = val[~val_cold], test[~test_cold]

head = head_items_from_train(train)

print()
describe(train, "train")
describe(val, "val")
describe(test, "test")
print(f"head items: {len(head):,} ({len(head) / train['parent_asin'].nunique():.2%} of train items)")
print(f"test rows with rating >= {config.MIN_RATING}: {(test['rating'] >= config.MIN_RATING).mean():.1%}")

# ---------------------------------------------------------------------------
# 4. Save (names match config.TRAIN_PATH / VAL_PATH / TEST_PATH / HEAD_PATH)
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
info = {"category": CATEGORY, "k_core": K_CORE, "source": IN_PATH.name, "head_share": HEAD_SHARE}
for name, part in [("train", train), ("val", val), ("test", test), ("head_items", head)]:
    path = OUT_DIR / f"{name}.parquet"
    part.reset_index(drop=True).to_parquet(path, index=False)
    info[name] = {"rows": len(part), "md5": file_hash(path)}
    if "user_id" in part:
        info[name].update(users=int(part["user_id"].nunique()),
                          items=int(part["parent_asin"].nunique()))

(OUT_DIR / "split_info.json").write_text(json.dumps(info, indent=2))
print("\nSaved to", OUT_DIR)
print(json.dumps(info, indent=2))
