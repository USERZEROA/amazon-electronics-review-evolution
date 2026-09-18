import polars as pl

from analysis import (
    clean_ratings,
    add_features,
    get_yearly_summary,
    get_monthly_summary,
    get_verified_comparison,
)


def test_small_analysis_pipeline(tmp_path):
    data = pl.DataFrame({
        "rating": [
            0.0,
            5.0,
            4.0,
            2.0,
            1.0,
            6.0,
        ],
        "timestamp": [
            1640995200000,
            1640995200000,
            1640995200000,
            1643673600000,
            1672531200000,
            1675209600000,
        ],
        "verified_purchase": [
            True,
            True,
            False,
            True,
            False,
            True,
        ],
        "review_length_chars": [
            50,
            100,
            200,
            300,
            400,
            500,
        ],
        "helpful_vote": [
            0,
            1,
            2,
            0,
            3,
            4,
        ],
        "user_id": [
            "u0",
            "u1",
            "u2",
            "u3",
            "u4",
            "u5",
        ],
        "parent_asin": [
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
        ],
    })

    file_path = tmp_path / "reviews.parquet"
    data.write_parquet(file_path)

    reviews = pl.scan_parquet(file_path)

    reviews = clean_ratings(reviews)
    reviews = add_features(reviews)

    cleaned = reviews.collect()

    assert cleaned.height == 4
    assert cleaned["rating"].min() == 1.0
    assert cleaned["rating"].max() == 5.0

    assert "year" in cleaned.columns
    assert "month" in cleaned.columns
    assert "is_positive" in cleaned.columns

    yearly = get_yearly_summary(
        reviews,
        start_year=2022,
        end_year=2023,
    )

    monthly = get_monthly_summary(
        reviews,
        start_month="2022-01",
        end_month="2023-03",
    )

    verified = get_verified_comparison(
        reviews,
        start_month="2022-01",
        end_month="2023-03",
    )

    row_2022 = (
        yearly
        .filter(pl.col("year") == 2022)
        .row(0, named=True)
    )

    row_2023 = (
        yearly
        .filter(pl.col("year") == 2023)
        .row(0, named=True)
    )

    assert row_2022["review_count"] == 3
    assert row_2023["review_count"] == 1

    assert monthly["review_count"].sum() == 4
    assert verified["review_count"].sum() == 4