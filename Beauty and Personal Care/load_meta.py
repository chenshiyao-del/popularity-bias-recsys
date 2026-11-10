import pandas as pd

# change path for other categories 
path = "data/interim/Beauty_and_Personal_Care_meta.parquet"

df = pd.read_parquet(
    path,
    columns=[
        "parent_asin",
        "main_category",
        "store",
        "price",
        "average_rating",
        "rating_number",
    ],
)

print("shape:", df.shape)
print("dtypes:\n", df.dtypes)
print("head:\n", df.head(5))