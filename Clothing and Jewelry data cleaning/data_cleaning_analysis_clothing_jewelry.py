import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


CATEGORY = "Clothing_Shoes_and_Jewelry"

REPO_DIR = Path(__file__).resolve().parent.parent
INTERIM = REPO_DIR / "data" / "interim"
REVIEWS_PATH = INTERIM / f"{CATEGORY}_full.parquet"
META_PATH = INTERIM / f"{CATEGORY}_meta.parquet"
OUT = INTERIM / f"{CATEGORY}_interactions.parquet"

COLUMNS = ["rating", "asin", "parent_asin", "user_id",
           "timestamp", "verified_purchase"]

META_FIELDS = ["main_category", "average_rating", "rating_number", "price",
               "store", "categories", "details", "parent_asin"]

ONLY_VERIFIED = True
MAX_ROW_GROUPS = None   # set to e.g. 2 for a quick test run



# ---------------------------------------------------------------------------
# Reviews: read every row group in chunks, clean each chunk
# ---------------------------------------------------------------------------
pf = pq.ParquetFile(REVIEWS_PATH)
n_groups = pf.num_row_groups if MAX_ROW_GROUPS is None else min(MAX_ROW_GROUPS, pf.num_row_groups)

chunks = []
raw_rows = 0
out_of_range = 0
verified_counts = pd.Series(dtype="int64")
rating_sums = pd.Series(dtype="float64")
rating_counts = pd.Series(dtype="int64")

for i in range(n_groups):
    chunk = pf.read_row_group(i, columns=COLUMNS).to_pandas()
    raw_rows += len(chunk)

    # Fix types (bad values become NaN/NaT) and drop rows with missing values
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms", errors="coerce")
    chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
    chunk = chunk.dropna(subset=COLUMNS)
    out_of_range += (~chunk["rating"].between(1, 5)).sum()

    # Verified-purchase stats, accumulated before filtering
    verified_counts = verified_counts.add(chunk["verified_purchase"].value_counts(), fill_value=0)
    grp = chunk.groupby("verified_purchase")["rating"]
    rating_sums = rating_sums.add(grp.sum(), fill_value=0)
    rating_counts = rating_counts.add(grp.count(), fill_value=0)

    # Verified filter BEFORE dedup, so a later unverified review can't replace a verified one
    if ONLY_VERIFIED:
        chunk = chunk[chunk["verified_purchase"] == True]

    # Keep only interaction columns, with smaller dtypes to save memory
    chunk = chunk[["user_id", "parent_asin", "rating", "timestamp"]].copy()
    chunk["rating"] = chunk["rating"].astype("float32")
    chunk["user_id"] = chunk["user_id"].astype("string[pyarrow]")
    chunk["parent_asin"] = chunk["parent_asin"].astype("string[pyarrow]")

    chunks.append(chunk)
    print(f"row group {i + 1}/{n_groups} -> kept {len(chunk):,} rows", flush=True)

reviews = pd.concat(chunks, ignore_index=True)
del chunks

print(f"\nRaw rows read: {raw_rows:,}")
print("Ratings outside 1-5:", out_of_range)
print("Verified share:\n", verified_counts / verified_counts.sum())
print(pd.DataFrame({"mean": rating_sums / rating_counts,
                    "count": rating_counts.astype("int64")}))
print("Rows kept after verified filter:", f"{len(reviews):,}")

# Dedup across ALL chunks (a user's duplicates can span row groups), keep latest review
n0 = len(reviews)
interactions = (reviews.sort_values("timestamp")
                       .drop_duplicates(["user_id", "parent_asin"], keep="last")
                       .reset_index(drop=True))
del reviews
print(f"Duplicate (user, parent_asin) rows removed: {n0 - len(interactions):,}")

interactions.to_parquet(OUT, index=False)
print("Saved to:", OUT)


# ---------------------------------------------------------------------------
# Sanity checks on the final interactions
# ---------------------------------------------------------------------------
n_users = interactions["user_id"].nunique()
n_items = interactions["parent_asin"].nunique()
print("\nrows:", f"{len(interactions):,}")
print("users:", f"{n_users:,}")
print("items:", f"{n_items:,}")
print("density:", len(interactions) / (n_users * n_items))
print("rating distribution:\n", interactions["rating"].value_counts(normalize=True).sort_index())
print("time range:", interactions["timestamp"].min(), "to", interactions["timestamp"].max())

user_counts = interactions["user_id"].value_counts()
print("\nuser interactions: min/median/max =",
      user_counts.min(), user_counts.median(), user_counts.max())

item_counts = interactions["parent_asin"].value_counts()
print("item interactions: min/median/max =",
      item_counts.min(), item_counts.median(), item_counts.max())


# ---------------------------------------------------------------------------
# Metadata: read every row group, keep only items in the final interactions
# ---------------------------------------------------------------------------
final_items = interactions["parent_asin"].unique()

mpf = pq.ParquetFile(META_PATH)
meta_chunks = []
for i in range(mpf.num_row_groups):
    m = mpf.read_row_group(i, columns=META_FIELDS).to_pandas()
    meta_chunks.append(m[m["parent_asin"].isin(final_items)])
meta = pd.concat(meta_chunks, ignore_index=True)
del meta_chunks

meta = meta.drop_duplicates("parent_asin").reset_index(drop=True)

# Flattened fields cleaning
meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
meta["store"] = meta["store"].replace("", np.nan)

# Item features (feature engineering; keep only if your models use them)
meta["brand"] = meta["details"].apply(
    lambda d: d.get("Brand") or d.get("brand") if isinstance(d, dict) else None)
meta["n_categories"] = meta["categories"].apply(lambda c: len(c) if isinstance(c, list) else 0)

# Coverage check on the final item set
coverage = pd.Series(final_items).isin(meta["parent_asin"]).mean()
print(f"\nFinal items with metadata: {coverage:.2%}")
print("Items in interactions:", f"{len(final_items):,}")
print("Items in meta:", f"{len(meta):,}")
print("Meta missing share:\n", meta[["main_category", "price", "store", "brand"]].isna().mean()) 
