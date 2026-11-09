import gzip
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache\datasets"


SCRIPT_DIR = Path(__file__).resolve().parent   
REPO_DIR = SCRIPT_DIR.parent                   
PROJECT_DIR = REPO_DIR.parent                  

RAW_DIR = PROJECT_DIR
OUT_DIR = REPO_DIR / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

category = "Amazon_Fashion"
src_path = RAW_DIR / f"{category}.jsonl"
out_path = OUT_DIR / f"{category}_full.parquet"

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


SCHEMA = pa.schema([
    ("rating", pa.float64()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("asin", pa.string()),
    ("parent_asin", pa.string()),
    ("user_id", pa.string()),
    ("timestamp", pa.int64()),
    ("verified_purchase", pa.bool_()),
])


def write_batch(records, writer):
    """Convert a list of dicts to a typed table and append it to the Parquet file."""
    df = pd.DataFrame(records, columns=REVIEW_FIELDS)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["verified_purchase"] = df["verified_purchase"].astype("boolean")

    # Batch matching with schema
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    writer.write_table(table)
    return len(df)


FLUSH_SIZE = 500_000
records = []
total_rows = 0
chunk_id = 0


def open_text(path):
    """Open a .jsonl file as text, transparently handling gzip or plain text."""
    with open(path, "rb") as fb:
        is_gzip = fb.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


writer = pq.ParquetWriter(out_path, SCHEMA)

try:
    with open_text(src_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                records.append({k: obj.get(k) for k in REVIEW_FIELDS})
            except json.JSONDecodeError:
                continue

            if len(records) >= FLUSH_SIZE:
                total_rows += write_batch(records, writer)
                chunk_id += 1
                records = []
                print(f"[chunk {chunk_id}] flushed, total rows = {total_rows}")

    if records:
        total_rows += write_batch(records, writer)
finally:
    writer.close()

print(f"[done] saved to {out_path}, total rows = {total_rows}")