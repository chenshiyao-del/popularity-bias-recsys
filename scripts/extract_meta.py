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
src_path = os.path.join(RAW_DIR, f"meta_{category}.jsonl", f"meta_{category}.jsonl")
out_path = os.path.join(OUT_DIR, f"{category}_meta.parquet")

META_FIELDS = [
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "store",
    "categories",
    "details",
    "parent_asin",
    "bought_together",
]

JSON_FIELDS = {"features", "description", "categories", "details", "bought_together"}

SCHEMA = pa.schema([
    pa.field("main_category", pa.string()),
    pa.field("title", pa.string()),
    pa.field("average_rating", pa.float64()),
    pa.field("rating_number", pa.int64()),
    pa.field("features", pa.string()),
    pa.field("description", pa.string()),
    pa.field("price", pa.float64()),
    pa.field("store", pa.string()),
    pa.field("categories", pa.string()),
    pa.field("details", pa.string()),
    pa.field("parent_asin", pa.string()),
    pa.field("bought_together", pa.string()),
])

FLUSH_SIZE = 100_000
records = []
writer = None
total_rows = 0
chunk_id = 0


def to_json_str(v):
    if v is None:
        return None
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def parse_price(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def clean_record(obj):
    rec = {}
    for k in META_FIELDS:
        v = obj.get(k)
        if k in JSON_FIELDS and v is not None:
            rec[k] = to_json_str(v)
        elif k == "price":
            rec[k] = parse_price(v)
        elif k == "average_rating":
            try:
                rec[k] = float(v) if v is not None else None
            except (TypeError, ValueError):
                rec[k] = None
        elif k == "rating_number":
            try:
                rec[k] = int(v) if v is not None else None
            except (TypeError, ValueError):
                rec[k] = None
        else:
            rec[k] = v
    return rec


def flush(records):
    df = pd.DataFrame(records, columns=META_FIELDS)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    return table


with open(src_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        records.append(clean_record(obj))

        if len(records) >= FLUSH_SIZE:
            table = flush(records)
            if writer is None:
                writer = pq.ParquetWriter(out_path, SCHEMA)
            writer.write_table(table)

            total_rows += len(records)
            chunk_id += 1
            records = []
            print(f"[chunk {chunk_id}] flushed, total rows = {total_rows}")

if records:
    table = flush(records)
    if writer is None:
        writer = pq.ParquetWriter(out_path, SCHEMA)
    writer.write_table(table)
    total_rows += len(records)
    chunk_id += 1
    print(f"[chunk {chunk_id}] flushed final, total rows = {total_rows}")

if writer is not None:
    writer.close()

print(f"[done] saved to {out_path}, total rows = {total_rows}")