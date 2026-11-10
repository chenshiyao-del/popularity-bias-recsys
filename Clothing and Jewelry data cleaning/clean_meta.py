import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
# change path for other categories
path = "data/interim/Clothing_Shoes_and_Jewelry_meta.parquet"

COLUMNS = ["rating", "title", "text", "asin", "parent_asin", "user_id",
           "timestamp", "verified_purchase"]
meta = pq.ParquetFile(path).read_row_group(0, columns=COLUMNS).to_pandas().sample(n=10_000, random_state=42).reset_index(drop=True)


# Getting real datetime and fixing corrupted ratings
meta["timestamp"] = pd.to_datetime(meta["timestamp"], unit="ms")
meta["rating"] = pd.to_numeric(meta["rating"], errors="coerce")
meta = meta.dropna(subset=COLUMNS)

# Removing duplicates
n0 = len(meta)
meta = meta.drop_duplicates()
print(f"Exact duplicates removed: {n0 - len(meta):,}")
dup_pairs = meta.duplicated(["user_id", "parent_asin"], keep=False).sum()
print(f"Rows in repeated (user, parent_asin) pairs: {dup_pairs:,} ({dup_pairs/len(meta):.2%})")

meta = (meta.sort_values("timestamp")
                  .drop_duplicates(["user_id", "parent_asin"], keep="last")
                  .reset_index(drop=True))

# Verified purchase cleaning
print(meta["verified_purchase"].value_counts(normalize=True))
print(meta.groupby("verified_purchase")["rating"].agg(["mean", "count"]))
ONLY_VERIFIED = True
if ONLY_VERIFIED:
    meta = meta[meta["verified_purchase"] == True].copy()
print("Rows kept:", len(meta))

POSITIVE_THRESHOLD = None
interactions = meta.copy()
interactions = interactions[["user_id", "parent_asin", "rating", "timestamp"]]
print("Interactions:", len(interactions))

# Checking to make sure all the cleaning worked.
print("rows:", len(interactions))
print("users:", interactions["user_id"].nunique())
print("items:", interactions["parent_asin"].nunique())
print("density:", len(interactions) / (interactions["user_id"].nunique() * interactions["parent_asin"].nunique()))
print("rating distribution:\n", interactions["rating"].value_counts(normalize=True).sort_index())
print("time range:", interactions["timestamp"].min(), "to", interactions["timestamp"].max())

# users' interaction counts
user_counts = interactions["user_id"].value_counts()
print("\nuser interactions: min/median/max =",
      user_counts.min(), user_counts.median(), user_counts.max())

# items' interaction counts
item_counts = interactions["parent_asin"].value_counts()
print("item interactions: min/median/max =",
      item_counts.min(), item_counts.median(), item_counts.max())
