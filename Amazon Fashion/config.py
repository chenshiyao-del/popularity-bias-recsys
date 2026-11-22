# src/config.py
from pathlib import Path

# ---------- Configuration ----------
REPO_DIR = Path(__file__).resolve().parent.parent
PROC = REPO_DIR / "data" / "processed"
CATEGORY = "Amazon_Fashion"
K_CORE = 2

K = 10                          # length of recommendation list (Recall@10)
D = 2000                        # candidate depth: None means ranking over all unseen items
CHUNK = 100_000                 # users processed per batch (keeps memory low)

# Path
DATA_DIR = PROC / CATEGORY / f"k{K_CORE}"
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
HEAD_PATH = DATA_DIR / "head_items.parquet"   # written by split.py, checked by leakage_guard (R4)

MIN_RATING = 3.0