import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import pyarrow.parquet as pq


REPO_DIR = Path(__file__).resolve().parent.parent
path = REPO_DIR / "data" / "interim" / "Amazon_Fashion_full.parquet"
reviews = pd.read_parquet(path, columns = ["rating", "title", "text", "asin", "parent_asin", "user_id",
               "timestamp", "verified_purchase"])

COLUMNS = ["rating", "title", "text", "asin", "parent_asin", "user_id",
           "timestamp", "verified_purchase"]

SAMPLE_FRACTION = 0.25
RANDOM_SEED = 42
df = pq.ParquetFile(path).read_row_group(0, columns=COLUMNS).to_pandas()
users = df["user_id"].drop_duplicates().sample(frac=1, random_state=RANDOM_SEED)
keep = df["user_id"].value_counts().reindex(users).cumsum()
keep = keep[keep <= 10_000].index
reviews = df[df["user_id"].isin(keep)].reset_index(drop=True)

#reviews = pq.ParquetFile(path).read_row_group(0, columns=COLUMNS).to_pandas().sample(n=10_000, random_state=42).reset_index(drop=True)

REPO_DIR = Path(__file__).resolve().parent.parent
meta_path = REPO_DIR / "data" / "interim" / "Amazon_Fashion_meta.parquet"

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
    "user_id"
]
#meta = (pq.ParquetFile(meta_path).read_row_group(0, columns=META_FIELDS).to_pandas().sample(n=10_000, random_state=RANDOM_SEED).reset_index(drop=True))
meta = pq.ParquetFile(meta_path).read_row_group(0, columns=META_FIELDS).to_pandas()
meta = meta[meta["parent_asin"].isin(reviews["parent_asin"])]
# Handling missing values, converting datatypes for timestamp and rating
reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="ms")
reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")

print("Date range:", reviews["timestamp"].min(), "->", reviews["timestamp"].max())
print("Ratings outside 1-5:", (~reviews["rating"].between(1, 5)).sum())
print("Missing values:\n", reviews.isna().sum())
print("Empty review text:", (reviews["text"].fillna("").str.strip() == "").sum())
reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"])


# Removing duplicate user item interactions
n0 = len(reviews)
reviews = reviews.drop_duplicates()   # exact duplicate rows
dup_pairs = reviews.duplicated(["user_id", "parent_asin"], keep=False).sum()
print(f"Exact duplicates removed: {n0 - len(reviews):,}")
print(f"Rows in repeated (user, parent_asin) pairs: {dup_pairs:,} ({dup_pairs/len(reviews):.2%})")

reviews = (reviews.sort_values("timestamp")
                  .drop_duplicates(["user_id", "parent_asin"], keep="last")
                  .reset_index(drop=True))
print("After dedup:", len(reviews))


print(reviews["verified_purchase"].value_counts(normalize=True))
print(reviews.groupby("verified_purchase")["rating"].agg(["mean", "count"]))

ONLY_VERIFIED = True   
if ONLY_VERIFIED:
    reviews = reviews[reviews["verified_purchase"] == True].copy()
print("Rows kept:", len(reviews))


interactions = reviews.copy()
interactions = interactions[["user_id", "parent_asin", "rating", "timestamp",
                             "verified_purchase", "text"]]
print("Interactions:", len(interactions))

meta = meta.drop_duplicates("parent_asin")
meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
meta["store"] = meta["store"].replace("", np.nan)
meta["brand"] = meta["details"].apply(
    lambda d: d.get("Brand") or d.get("brand") if isinstance(d, dict) else None)
meta["n_categories"] = meta["categories"].apply(lambda c: len(c) if isinstance(c, list) else 0)

print("Meta missing share:\n", meta[["main_category", "price", "store", "brand"]].isna().mean())

reviewed_items = reviews["parent_asin"].unique()
coverage = np.isin(reviewed_items, meta["parent_asin"]).mean()
print(f"Reviewed items with metadata: {coverage:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
reviews["rating"].value_counts().sort_index().plot.bar(ax=axes[0], title="Rating distribution")
sns.histplot(reviews.groupby("parent_asin")["rating"].mean(), bins=40, ax=axes[1])
axes[1].set_title("Mean rating per item")
plt.tight_layout(); plt.show()

data = interactions.copy()   


if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", errors="coerce")


meta_items = meta.drop_duplicates("parent_asin").set_index("parent_asin")
meta_items["store"] = meta_items["store"].replace("", np.nan)
# description / features were saved as JSON strings; an empty list "[]" counts as missing
for col in ["description", "features"]:
    if col in meta_items.columns:
        meta_items[col] = meta_items[col].replace("[]", np.nan)


# Defining the helper functions
def gini(x):
    """0 = all items equally popular, 1 = one item gets everything."""
    x = np.sort(np.asarray(x, dtype=float))
    n, cum = len(x), np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def head_tail(df, head_share=0.20):
    """Head = most popular items that together account for `head_share` of interactions."""
    counts = df["parent_asin"].value_counts()            # sorted descending
    cum = counts.cumsum() / counts.sum()
    head = set(cum[cum <= head_share].index) or {counts.index[0]}
    return counts, head


def lorenz(counts):
    x = np.sort(counts.values)
    y = np.insert(np.cumsum(x) / x.sum(), 0, 0)
    return np.linspace(0, 1, len(y)), y


def summarize(df, label=""):
    n_u, n_i, n = df["user_id"].nunique(), df["parent_asin"].nunique(), len(df)
    return pd.Series({"interactions": n, "users": n_u, "items": n_i,
                      "density": n / (n_u * n_i) if n_u and n_i else np.nan,
                      "avg_per_user": n / n_u if n_u else np.nan,
                      "avg_per_item": n / n_i if n_i else np.nan}, name=label)

print(summarize(data, "raw").to_frame())

# Users activity and cold start

u_cnt = data["user_id"].value_counts()
i_cnt = data["parent_asin"].value_counts()

print(f"Users with 1 review:  {(u_cnt == 1).mean():.1%}")
print(f"Items with 1 review:  {(i_cnt == 1).mean():.1%}")
print(f"Interactions on 1-review items: {i_cnt[i_cnt == 1].sum() / len(data):.1%}")
print("\nReviews per user:\n", u_cnt.describe(percentiles=[.5, .75, .9, .99]))
print("\nReviews per item:\n", i_cnt.describe(percentiles=[.5, .75, .9, .99]))

print(f"Users with exactly 1 interaction: {(u_cnt == 1).mean():.2%}")

ax = u_cnt.clip(upper=50).value_counts().sort_index().plot.bar(figsize=(11, 3.5))
ax.set(title="Interactions per user (capped at 50)", xlabel="# interactions", ylabel="# users")
plt.tight_layout(); plt.show()


# Long tail: item popularity, Lorenz curve, Gini
item_counts, HEAD_ITEMS = head_tail(data)   
print(f"Gini = {gini(item_counts):.3f} | head items = {len(HEAD_ITEMS):,} "
      f"({len(HEAD_ITEMS)/len(item_counts):.2%} of items -> 20% of interactions)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].loglog(np.arange(1, len(item_counts) + 1), item_counts.values)
axes[0].set(title="Item popularity (log-log)", xlabel="item rank", ylabel="# interactions")
axes[1].plot(*lorenz(item_counts), label="raw")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
axes[1].set(title="Lorenz curve", xlabel="cumulative share of items",
            ylabel="cumulative share of interactions")
axes[1].legend()
plt.tight_layout(); plt.show()


# User profiles: mainstream / diverse / niche (only users with ≥ 3 reviews are profiled; items are not filtered)

MIN_USER_REVIEWS = 2   

user_profile = (data.assign(item_pop=data["parent_asin"].map(item_counts),
                            is_tail=~data["parent_asin"].isin(HEAD_ITEMS))
                    .groupby("user_id")
                    .agg(n=("item_pop", "size"),
                         avg_item_pop=("item_pop", "mean"),
                         tail_ratio=("is_tail", "mean")))


profiled = user_profile[user_profile["n"] >= MIN_USER_REVIEWS].copy()
print(f"Profiling {len(profiled):,} of {len(user_profile):,} users ...")

if len(profiled) >= 3:
    profiled["group"] = pd.qcut(profiled["tail_ratio"].rank(method="first"), 3,
                                labels=["mainstream", "diverse", "niche"])
    print(profiled.groupby("group", observed=True)[["avg_item_pop", "tail_ratio"]].mean())
    sns.histplot(profiled["tail_ratio"], bins=30)
    plt.show()
else:
    print("Not enough multi-review users in this sample to profile.")


# Checking whether head or tail items look the same or different
agg_spec = dict(mean_rating=("rating", "mean"),
                verified_share=("verified_purchase", "mean"),
                n=("rating", "size"))
tmp = data.assign(is_head=data["parent_asin"].isin(HEAD_ITEMS))
if "text" in tmp.columns:
    tmp["review_len"] = tmp["text"].fillna("").str.split().str.len()
    agg_spec["median_review_len"] = ("review_len", "median")

print(tmp.groupby("is_head").agg(**agg_spec))
del tmp

# Time: interactions and concentration per year
yearly = data.groupby(data["timestamp"].dt.year).agg(
    interactions=("user_id", "size"),
    gini=("parent_asin", lambda s: gini(s.value_counts())))
print(yearly)
yearly["interactions"].plot.bar(figsize=(10, 3.5), title="Interactions per year")
plt.tight_layout(); plt.show()

# Item arrival over time
first_seen = data.groupby("parent_asin")["timestamp"].min().dt.year.rename("first_year")
is_head_item = pd.Series(first_seen.index.isin(HEAD_ITEMS), index=first_seen.index, name="is_head")

arrival = pd.crosstab(first_seen, is_head_item)
arrival["new_items"] = arrival.sum(axis=1)
arrival["head_share"] = arrival.get(True, 0) / arrival["new_items"]
print(arrival)

# Store concentration
m = data[["parent_asin"]].join(meta_items[["store"]], on="parent_asin", how="left")
store_counts = m["store"].dropna().value_counts()
top = max(1, int(len(store_counts) * 0.02))
print(f"Top 2% of stores ({top:,}) account for "
      f"{store_counts.iloc[:top].sum()/store_counts.sum():.2%} of reviews")
print(f"Store Gini: {gini(store_counts):.3f} | reviews with no store: {m['store'].isna().mean():.2%}")


# Metadata coverage over head and tail
item_meta = meta_items.reindex(item_counts.index)
item_meta["is_head"] = item_meta.index.isin(HEAD_ITEMS)
item_meta["in_meta"] = item_meta.index.isin(meta_items.index)

cov_cols = [c for c in ["store", "price", "description", "features"] if c in item_meta.columns]
print("Share of items MISSING each field:")
print(item_meta.groupby("is_head")[cov_cols].apply(lambda g: g.isna().mean()).round(3))
print("\nShare of items with any metadata row:")
print(item_meta.groupby("is_head")["in_meta"].mean().round(3))

# Price vs. popularity
item_stats = data.groupby("parent_asin")["rating"].agg(n="count", mean_rating="mean")
pp = item_stats[["n"]].join(meta_items["price"]).dropna(subset=["price"])
print(f"Items with a price: {len(pp)/len(item_stats):.2%}")

if len(pp) > 10:
    rho, p = spearmanr(pp["n"], pp["price"])
    print(f"Spearman(popularity, price) = {rho:.3f} (p={p:.2g})")
    pp["price_bin"] = pd.qcut(pp["price"], 5, duplicates="drop")
    print(pp.groupby("price_bin", observed=True)["n"].agg(["median", "mean", "count"]))