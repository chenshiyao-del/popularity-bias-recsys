"""Clean raw Amazon reviews into a table of (user, item, rating, timestamp) interactions.

Keeps verified purchases only and, when a user reviewed the same item more than once, only their
latest review. Writes data/<category>_interactions.parquet.

Usage: python prepare.py <reviews.jsonl> <category>
"""
import gzip
import json
import os
import sys
import pandas as pd

path, cat = sys.argv[1], sys.argv[2]

# Load raw reviews
print("Streaming reading file...")
records = []
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            # only extract 5 fields 
            records.append({
                "user_id": obj.get("user_id"),
                "parent_asin": obj.get("parent_asin"),
                "rating": obj.get("rating"),
                "timestamp": obj.get("timestamp"),
                "verified_purchase": obj.get("verified_purchase")
            })
        except json.JSONDecodeError:
            continue

print(f"Streamed {len(records):,} rows. Converting to DataFrame...")
df = pd.DataFrame(records)
del records 

# Clean: verified purchases, needed columns, one interaction per (user, item)
df = df[df["verified_purchase"] == True]
df = df[["user_id", "parent_asin", "rating", "timestamp"]].dropna()
df = df.sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "parent_asin"], keep="last")

# Save
os.makedirs("data", exist_ok=True)
df.to_parquet(f"data/{cat}_interactions.parquet", index=False)
print(f"{len(df):,} interactions | {df.user_id.nunique():,} users | {df.parent_asin.nunique():,} items")
