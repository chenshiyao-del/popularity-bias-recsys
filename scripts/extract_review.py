import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache\datasets"

# raw files live in "Data cleaning/<name>.jsonl/<name>.jsonl", two levels above this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = BASE_DIR
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "interim")
os.makedirs(OUT_DIR, exist_ok=True)

category = "Clothing_Shoes_and_Jewelry"
src_path = os.path.join(RAW_DIR, f"{category}.jsonl", f"{category}.jsonl")
out_path = os.path.join(OUT_DIR, f"{category}_full.parquet")

REVIEW_FIELDS = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "verified_purchase",
]

records = []
FLUSH_SIZE = 500_000
writer = None
total_rows = 0
chunk_id = 0

with open(src_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            records.append({k: obj.get(k) for k in REVIEW_FIELDS})
        except json.JSONDecodeError:
            continue

        if len(records) >= FLUSH_SIZE:
            df = pd.DataFrame(records, columns=REVIEW_FIELDS)

            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df["verified_purchase"] = df["verified_purchase"].astype("boolean")

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            total_rows += len(df)
            chunk_id += 1
            records = []
            print(f"[chunk {chunk_id}] flushed, total rows = {total_rows}")

if records:
    df = pd.DataFrame(records, columns=REVIEW_FIELDS)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["verified_purchase"] = df["verified_purchase"].astype("boolean")

    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    total_rows += len(df)

if writer is not None:
    writer.close()

print(f"[done] saved to {out_path}, total rows = {total_rows}")