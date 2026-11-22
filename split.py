"""Filter the interactions to a k-core and split them into train / validation / test by time.

k-core: repeatedly drop users and items with fewer than k interactions.
Leave-last-out: each user's latest interaction is test (users with >= 2), the one before it is
validation (users with >= 3), and the rest is train. Rows are written sorted by user, then time.
Writes data/<category>_k<k>_{train,val,test}.parquet.

Usage: python split.py <category> <k>
"""
import sys
import pandas as pd

cat, k = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(f"data/{cat}_interactions.parquet")

# k-core filter
while True:
    n = len(df)
    df = df[df.groupby("parent_asin")["user_id"].transform("size") >= k]
    df = df[df.groupby("user_id")["parent_asin"].transform("size") >= k]
    if len(df) == n:
        break

# Leave-last-out split per user
df = df.sort_values(["user_id", "timestamp", "parent_asin"], kind="stable")
pos = df.groupby("user_id").cumcount(ascending=False)
n_user = df.groupby("user_id")["user_id"].transform("size")
is_test, is_val = (pos == 0) & (n_user >= 2), (pos == 1) & (n_user >= 3)
train, val, test = df[~is_test & ~is_val], df[is_val], df[is_test]

# Drop validation/test items that never appear in train
val = val[val.parent_asin.isin(train.parent_asin)]
test = test[test.parent_asin.isin(train.parent_asin)]

# Save
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_parquet(f"data/{cat}_k{k}_{name}.parquet", index=False)
    print(f"{name:>5}: {len(part):,} rows | {part.user_id.nunique():,} users | {part.parent_asin.nunique():,} items")
