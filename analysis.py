"""
Amazon Electronics Review Trends Around the Rise
of Generative AI
"""

from pathlib import Path
import time
from statistics import median

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Dataset Import
# ============================================================

ROOT = Path(__file__).resolve().parent

REVIEWS_PATH = ROOT / "data" / "processed" / "reviews.parquet"
ML_PATH = ROOT / "data" / "processed" / "reviews_ml_sample.parquet"

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

reviews_lf = pl.scan_parquet(REVIEWS_PATH)

print("Dataset loaded with Polars LazyFrame.")
print(reviews_lf.collect_schema())


# ============================================================
# 2. Data Inspection
# ============================================================

print("\n=== Full Dataset Overview ===")

overview = (
    reviews_lf
    .select(
        pl.len().alias("total_reviews"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating"),
        pl.col("rating").mean().alias("mean_rating"),

        pl.from_epoch("timestamp", time_unit="ms")
        .min()
        .alias("first_review"),

        pl.from_epoch("timestamp", time_unit="ms")
        .max()
        .alias("last_review"),

        pl.col("user_id")
        .n_unique()
        .alias("unique_users"),

        pl.col("parent_asin")
        .n_unique()
        .alias("unique_products"),
    )
    .collect()
)

print(overview)


invalid_ratings = (
    reviews_lf
    .filter(
        (pl.col("rating") < 1.0)
        | (pl.col("rating") > 5.0)
    )
    .group_by("rating")
    .agg(
        pl.len().alias("count")
    )
    .sort("rating")
    .collect()
)

print("\n=== Invalid Ratings ===")
print(invalid_ratings)


PANDAS_INSPECTION_ROWS = 200_000

pandas_sample = (
    reviews_lf
    .head(PANDAS_INSPECTION_ROWS)
    .collect()
    .to_pandas()
)

print("\n=== First Five Rows ===")
print(pandas_sample.head())

print("\n=== Pandas Info ===")
pandas_sample.info()

print("\n=== Summary Statistics ===")
print(pandas_sample.describe())

print("\n=== Missing Values ===")
print(pandas_sample.isna().sum())

print("\n=== Duplicate-Looking Rows ===")
print(pandas_sample.duplicated().sum())


# ============================================================
# 3. Data Cleaning and Feature Engineering
# ============================================================

reviews = (
    reviews_lf
    .filter(
        pl.col("rating").is_between(1.0, 5.0)
    )
    .with_columns(
        pl.from_epoch(
            "timestamp",
            time_unit="ms"
        ).alias("review_datetime")
    )
    .with_columns(
        pl.col("review_datetime")
        .dt.year()
        .alias("year"),

        pl.col("review_datetime")
        .dt.strftime("%Y-%m")
        .alias("month"),

        (pl.col("rating") >= 4.0)
        .alias("is_positive"),
    )
)


# ============================================================
# 4. Filtering and Grouping
# ============================================================

# Use complete years for the longer-term comparison.
yearly_reviews = reviews.filter(
    pl.col("year").is_between(2018, 2022)
)

yearly_summary = (
    yearly_reviews
    .group_by("year")
    .agg(
        pl.len().alias("review_count"),

        pl.col("rating")
        .mean()
        .alias("avg_rating"),

        pl.col("is_positive")
        .mean()
        .alias("positive_review_rate"),

        pl.col("verified_purchase")
        .mean()
        .alias("verified_purchase_rate"),

        pl.col("review_length_chars")
        .mean()
        .alias("avg_review_length"),

        pl.col("helpful_vote")
        .mean()
        .alias("avg_helpful_votes"),

        pl.col("user_id")
        .n_unique()
        .alias("unique_reviewers"),

        pl.col("parent_asin")
        .n_unique()
        .alias("unique_products"),
    )
    .sort("year")
    .collect()
)

print("\n=== Yearly Summary: 2018-2022 ===")

print(
    yearly_summary
    .to_pandas()
    .to_string(index=False)
)


# Focus on Jan 2022 through Mar 2023.
monthly_focus = (
    reviews
    .filter(
        (pl.col("month") >= "2022-01")
        & (pl.col("month") <= "2023-03")
    )
    .group_by("month")
    .agg(
        pl.len().alias("review_count"),

        pl.col("rating")
        .mean()
        .alias("avg_rating"),

        pl.col("review_length_chars")
        .mean()
        .alias("avg_review_length"),

        pl.col("verified_purchase")
        .mean()
        .alias("verified_purchase_rate"),
    )
    .sort("month")
    .collect()
)

print("\n=== Monthly Summary: Jan 2022 - Mar 2023 ===")

print(
    monthly_focus
    .to_pandas()
    .to_string(index=False)
)


# Check the 2023 monthly counts because later months
# contain far fewer reviews.
coverage_2023 = (
    reviews
    .filter(pl.col("year") == 2023)
    .group_by("month")
    .agg(
        pl.len().alias("review_count")
    )
    .sort("month")
    .collect()
)

print("\n=== 2023 Review Coverage Check ===")

print(
    coverage_2023
    .to_pandas()
    .to_string(index=False)
)


# Compare review length for verified and non-verified reviews.
verified_comparison = (
    reviews
    .filter(
        (pl.col("month") >= "2022-01")
        & (pl.col("month") <= "2023-03")
    )
    .group_by(
        "month",
        "verified_purchase"
    )
    .agg(
        pl.len().alias("review_count"),

        pl.col("review_length_chars")
        .mean()
        .alias("avg_review_length"),

        pl.col("rating")
        .mean()
        .alias("avg_rating"),
    )
    .sort(
        "month",
        "verified_purchase"
    )
    .collect()
)

print("\n=== Verified vs. Non-Verified Reviews ===")

verified_table = (
    verified_comparison
    .to_pandas()
)

print(
    verified_table
    .to_string(index=False)
)


# ============================================================
# 5. Pandas vs. Polars
# ============================================================

benchmark_rows = 5_000_000

benchmark_pl = (
    reviews
    .select(
        "year",
        "rating",
        "verified_purchase"
    )
    .head(benchmark_rows)
    .collect()
)

benchmark_pd = benchmark_pl.to_pandas()


# Warm-up run
benchmark_pd.groupby("year").agg(
    review_count=("rating", "size"),
    avg_rating=("rating", "mean"),
    verified_purchase_rate=("verified_purchase", "mean"),
)

benchmark_pl.group_by("year").agg(
    pl.len().alias("review_count"),
    pl.col("rating").mean().alias("avg_rating"),
    pl.col("verified_purchase").mean().alias("verified_purchase_rate"),
)


pandas_times = []
polars_times = []

for _ in range(5):

    start = time.perf_counter()

    pandas_result = (
        benchmark_pd
        .groupby("year")
        .agg(
            review_count=("rating", "size"),
            avg_rating=("rating", "mean"),
            verified_purchase_rate=("verified_purchase", "mean"),
        )
    )

    pandas_times.append(
        time.perf_counter() - start
    )


    start = time.perf_counter()

    polars_result = (
        benchmark_pl
        .group_by("year")
        .agg(
            pl.len().alias("review_count"),
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("verified_purchase").mean().alias("verified_purchase_rate"),
        )
        .sort("year")
    )

    polars_times.append(
        time.perf_counter() - start
    )


pandas_time = median(pandas_times)
polars_time = median(polars_times)

print("\n=== Pandas vs. Polars: 5 Million Rows ===")

print(
    f"Pandas median runtime: {pandas_time:.4f} seconds"
)

print(
    f"Polars median runtime: {polars_time:.4f} seconds"
)


# ============================================================
# 6. Visualization
# ============================================================

monthly_pd = monthly_focus.to_pandas()

monthly_pd["month"] = pd.to_datetime(
    monthly_pd["month"]
)


# Monthly review length around late 2022.
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_pd["month"],
    monthly_pd["avg_review_length"],
    marker="o",
)

plt.axvline(
    pd.Timestamp("2022-12-01"),
    linestyle="--",
)

plt.xlabel("Month")
plt.ylabel("Average Review Length (characters)")
plt.title("Amazon Electronics Review Length Around Late 2022")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "monthly_review_length.png",
    dpi=150,
)

plt.show()


# Longer-term yearly trend.
yearly_pd = yearly_summary.to_pandas()

plt.figure(figsize=(8, 5))

plt.plot(
    yearly_pd["year"],
    yearly_pd["avg_review_length"],
    marker="o",
)

plt.xlabel("Year")
plt.ylabel("Average Review Length (characters)")
plt.title("Average Amazon Electronics Review Length, 2018-2022")

plt.xticks(yearly_pd["year"])
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "review_length_by_year.png",
    dpi=150,
)

plt.show()


# Compare verified and non-verified review length.
verified_plot = verified_table.copy()

verified_plot["month"] = pd.to_datetime(
    verified_plot["month"]
)

verified_reviews = verified_plot[
    verified_plot["verified_purchase"] == True
]

non_verified_reviews = verified_plot[
    verified_plot["verified_purchase"] == False
]

plt.figure(figsize=(10, 5))

plt.plot(
    verified_reviews["month"],
    verified_reviews["avg_review_length"],
    marker="o",
    label="Verified purchase",
)

plt.plot(
    non_verified_reviews["month"],
    non_verified_reviews["avg_review_length"],
    marker="o",
    label="Non-verified purchase",
)

plt.xlabel("Month")
plt.ylabel("Average Review Length (characters)")
plt.title("Review Length by Verified Purchase Status")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "verified_review_length.png",
    dpi=150,
)

plt.show()


# ============================================================
# 7. Machine Learning Exploration
# ============================================================

print("\n=== Machine Learning: Review Sentiment ===")

ml_df = pd.read_parquet(
    ML_PATH,
    columns=[
        "text",
        "is_positive",
    ],
)

ml_df = ml_df.dropna(
    subset=["text"]
)

ml_df = ml_df[
    ml_df["text"].str.strip() != ""
]

print("ML rows:", len(ml_df))

print("\nClass counts:")

print(
    ml_df["is_positive"]
    .value_counts()
)


X = ml_df["text"]

y = (
    ml_df["is_positive"]
    .astype(int)
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


vectorizer = TfidfVectorizer(
    max_features=10_000,
    min_df=5,
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
)

model.fit(
    X_train_tfidf,
    y_train
)


predictions = model.predict(
    X_test_tfidf
)


print("\nAccuracy:")

print(
    accuracy_score(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive",
        ],
    )
)


words = vectorizer.get_feature_names_out()
weights = model.coef_[0]

positive_words = words[
    weights.argsort()[-10:][::-1]
]

negative_words = words[
    weights.argsort()[:10]
]


print("\nWords most associated with positive reviews:")
print(positive_words)

print("\nWords most associated with negative reviews:")
print(negative_words)


ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=[
        "Negative",
        "Positive",
    ],
    values_format="d",
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "confusion_matrix.png",
    dpi=150,
)

plt.show()