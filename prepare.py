"""Clean raw Amazon reviews into a table of (user, item, rating, timestamp) interactions.

Keeps verified purchases only and, when a user reviewed the same item more than once, only their
latest review. Writes data/<category>_interactions.parquet.

Usage: python prepare.py <reviews.jsonl> <category>
"""
import os
import sys
import pandas as pd

path, cat = sys.argv[1], sys.argv[2]

# Load raw reviews
df = pd.read_json(path, lines=True, convert_dates=False)

# Fix types (bad values become NaN/NaT) and drop rows with any missing value
cols = ["rating", "asin", "parent_asin", "user_id", "timestamp", "verified_purchase"]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset = cols)
df = df[df["verified_purchase"] == True]
df = df.sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "parent_asin"], keep="last")

# Save
os.makedirs("data", exist_ok=True)
df.to_parquet(f"data/{cat}_interactions.parquet", index=False)
print(f"{len(df):,} interactions | {df.user_id.nunique():,} users | {df.parent_asin.nunique():,} items")
