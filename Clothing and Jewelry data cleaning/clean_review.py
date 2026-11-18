import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Change path for other categories
path = Path(__file__).resolve().parent.parent / "interim" / "Clothing_Shoes_and_Jewelry_full.parquet"
OUT = Path(__file__).resolve().parent.parent / "interim" / "Clothing_Shoes_and_Jewelry_interactions.parquet"

KEEP = ["rating", "asin", "parent_asin", "user_id",
           "timestamp", "verified_purchase"]

ONLY_VERIFIED = True
MAX_ROW_GROUPS = None   # set to e.g. 2 for a quick test run


# The full file is ~9 GB / 66M rows, so it is read one row group at a time and
# only the interaction columns are carried forward.
pf = pq.ParquetFile(path)
n_groups = pf.num_row_groups if MAX_ROW_GROUPS is None else min(MAX_ROW_GROUPS, pf.num_row_groups)

chunks = []
raw_rows = 0
verified_counts = pd.Series(dtype="int64")
rating_sums = pd.Series(dtype="float64")
rating_counts = pd.Series(dtype="int64")

for i in range(n_groups):
    chunk = pf.read_row_group(i, columns=KEEP).to_pandas()
    raw_rows += len(chunk)

    # Getting real datetime and fixing corrupted ratings
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms", errors="coerce")
    chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
    chunk = chunk.dropna(subset=KEEP)

    # Verified purchase stats, accumulated across row groups
    verified_counts = verified_counts.add(chunk["verified_purchase"].value_counts(), fill_value=0)
    grp = chunk.groupby("verified_purchase")["rating"]
    rating_sums = rating_sums.add(grp.sum(), fill_value=0)
    rating_counts = rating_counts.add(grp.count(), fill_value=0)

    if ONLY_VERIFIED:
        chunk = chunk[chunk["verified_purchase"] == True]

    chunk = chunk[KEEP].copy()
    chunk["rating"] = chunk["rating"].astype("float32")
    chunk["user_id"] = chunk["user_id"].astype("string[pyarrow]")
    chunk["parent_asin"] = chunk["parent_asin"].astype("string[pyarrow]")
    chunks.append(chunk)
    print(f"row group {i + 1}/{n_groups} -> kept {len(chunk):,} rows", flush=True)

reviews = pd.concat(chunks, ignore_index=True)
del chunks
print("Raw rows read:", f"{raw_rows:,}")

print(verified_counts / verified_counts.sum())
print(pd.DataFrame({"mean": rating_sums / rating_counts, "count": rating_counts.astype("int64")}))
print("Rows kept:", len(reviews))

reviews = (reviews.sort_values("timestamp")
                  .drop_duplicates(["user_id", "parent_asin"], keep="last")
                  .reset_index(drop=True))

# Making every verified review a positive interaction
interactions = reviews[["user_id", "parent_asin", "rating", "timestamp"]]
print("Interactions:", len(interactions))

interactions.to_parquet(OUT, index=False)
print("Saved to:", OUT)

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
