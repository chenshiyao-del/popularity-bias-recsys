"""
Shared train / val / test split. Run ONCE; every model loads the saved files.
Nobody re-splits inside their own model code.

Input : data/interim/{CATEGORY}_interactions.parquet   (from the cleaning script)
Output: data/processed/{CATEGORY}/k{K_CORE}/
            train.parquet  val.parquet  test.parquet  split_info.json

Order of operations:
  1. (optional) k-core filter on the FULL interactions, before splitting
  2. temporal leave-last-out per user:
       latest interaction        -> test   (users with >= 2 interactions)
       second-latest interaction -> val    (users with >= 3 interactions)
       everything earlier        -> train
  3. drop val/test rows whose item never appears in train (cold items)

Usage:
  python split.py            # uses K_CORE below
  python split.py 5          # override, e.g. 5-core
"""
import sys
import json
import hashlib
import pandas as pd
from pathlib import Path

CATEGORY = "Beauty_and_Personal_Care"
K_CORE = 0            # 0 = no k-core filtering; 5 = every user and item has >= 5 interactions
if len(sys.argv) > 1:
    K_CORE = int(sys.argv[1])

REPO_DIR = Path(__file__).resolve().parent.parent
IN_PATH = REPO_DIR / "data" / "interim" / f"{CATEGORY}_interactions.parquet"
OUT_DIR = REPO_DIR / "data" / "processed" / CATEGORY / f"k{K_CORE}"


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
#    saw in training, so these rows would only add guaranteed misses.
# ---------------------------------------------------------------------------
train_items = set(train["parent_asin"].unique())
val_cold = ~val["parent_asin"].isin(train_items)
test_cold = ~test["parent_asin"].isin(train_items)
print(f"\nCold rows dropped: val {val_cold.sum():,} ({val_cold.mean():.1%}) | "
      f"test {test_cold.sum():,} ({test_cold.mean():.1%})")
val, test = val[~val_cold], test[~test_cold]

print()
describe(train, "train")
describe(val, "val")
describe(test, "test")
print(f"test rows with rating >= 3 (count as hits): {(test['rating'] >= 3).mean():.1%}")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
info = {"category": CATEGORY, "k_core": K_CORE, "source": IN_PATH.name}
for name, part in [("train", train), ("val", val), ("test", test)]:
    path = OUT_DIR / f"{name}.parquet"
    part.reset_index(drop=True).to_parquet(path, index=False)
    info[name] = {"rows": len(part), "users": int(part["user_id"].nunique()),
                  "items": int(part["parent_asin"].nunique()), "md5": file_hash(path)}

(OUT_DIR / "split_info.json").write_text(json.dumps(info, indent=2))
print("\nSaved to", OUT_DIR)
print(json.dumps(info, indent=2))