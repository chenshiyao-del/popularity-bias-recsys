import pandas as pd

# Change path for other categories 
path = "data/interim/Clothing_Shoes_and_Jewelry_full.parquet"

df = pd.read_parquet(
    path,
    columns=["user_id", "parent_asin", "rating", "timestamp"]
)

# convert timestape to seconds
df["timestamp"] = (df["timestamp"] // 1000).astype("int64")

print("rows:", len(df))
print("users:", df["user_id"].nunique())
print("items:", df["parent_asin"].nunique())
print("density:", len(df) / (df["user_id"].nunique() * df["parent_asin"].nunique()))
print("rating distribution:\n", df["rating"].value_counts(normalize=True).sort_index())
print("time range:", pd.to_datetime(df["timestamp"].min(), unit="s"),
      "to", pd.to_datetime(df["timestamp"].max(), unit="s"))

# users' interaction counts
user_counts = df["user_id"].value_counts()
print("\nuser interactions: min/median/max =",
      user_counts.min(), user_counts.median(), user_counts.max())

# items' interaction counts
item_counts = df["parent_asin"].value_counts()
print("item interactions: min/median/max =",
      item_counts.min(), item_counts.median(), item_counts.max())