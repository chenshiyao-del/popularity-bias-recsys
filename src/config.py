# src/config.py
from pathlib import Path

# ---------- Configuration ----------
REPO_DIR = Path(__file__).resolve().parent.parent
PROC = REPO_DIR / "data" / "processed" 
CATEGORY = "Beauty_and_Personal_Care"
K_CORE = 5                      

K = 10                          # length of recommendation list (Recall@10)
D = 2000                        # candidate depth: None means ranking over all unseen items
CHUNK = 100_000                 # users processed per batch (keeps memory low)

# Path
DATA_DIR = PROC / CATEGORY / f"k{K_CORE}"
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"

MIN_RATING = 3.0