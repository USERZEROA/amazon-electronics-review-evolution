
import pytest
import polars as pl

from analysis import (
    clean_ratings,
    add_features,
    get_yearly_summary,
    get_monthly_summary,
    get_verified_comparison,
)


def sample_reviews():
    return pl.DataFrame({
        "rating": [5.0, 3.0, 4.0, 2.0],
        "timestamp": [
            1640995200000,  # 2022-01-01
            1643673600000,  # 2022-02-01
            1672531200000,  # 2023-01-01
            1675209600000,  # 2023-02-01
        ],
        "verified_purchase": [True, False, True, False],
        "review_length_chars": [100, 300, 200, 400],
        "helpful_vote": [1, 2, 0, 3],
        "user_id": ["u1", "u2", "u3", "u3"],
        "parent_asin": ["p1", "p2", "p3", "p3"],
    })


def test_clean_ratings():
    df = pl.DataFrame({
        "rating": [0.0, 1.0, 3.0, 5.0, 6.0]
    })

    result = clean_ratings(df)

    assert result["rating"].to_list() == [1.0, 3.0, 5.0]


def test_add_features():
    df = sample_reviews()

    result = add_features(df)

    assert result["year"].to_list() == [
        2022, 2022, 2023, 2023
    ]

    assert result["month"].to_list() == [
        "2022-01",
        "2022-02",
        "2023-01",
        "2023-02",
    ]

    assert result["is_positive"].to_list() == [
        True,
        False,
        True,
        False,
    ]


def test_yearly_summary():
    df = sample_reviews()
    df = add_features(df)

    result = get_yearly_summary(
        df.lazy(),
        start_year=2022,
        end_year=2023,
    )

    row_2022 = (
        result
        .filter(pl.col("year") == 2022)
        .row(0, named=True)
    )

    assert row_2022["review_count"] == 2
    assert row_2022["avg_rating"] == pytest.approx(4.0)
    assert row_2022["positive_review_rate"] == pytest.approx(0.5)
    assert row_2022["verified_purchase_rate"] == pytest.approx(0.5)
    assert row_2022["avg_review_length"] == pytest.approx(200.0)
    assert row_2022["avg_helpful_votes"] == pytest.approx(1.5)
    assert row_2022["unique_reviewers"] == 2
    assert row_2022["unique_products"] == 2


def test_monthly_summary():
    df = sample_reviews()
    df = add_features(df)

    result = get_monthly_summary(
        df.lazy(),
        start_month="2022-02",
        end_month="2023-01",
    )

    assert result["month"].to_list() == [
        "2022-02",
        "2023-01",
    ]

    assert result["review_count"].to_list() == [1, 1]


def test_verified_comparison():
    df = pl.DataFrame({
        "month": [
            "2022-01",
            "2022-01",
            "2022-01",
        ],
        "verified_purchase": [
            True,
            True,
            False,
        ],
        "review_length_chars": [
            100,
            200,
            300,
        ],
        "rating": [
            5.0,
            4.0,
            2.0,
        ],
    })

    result = get_verified_comparison(
        df.lazy(),
        start_month="2022-01",
        end_month="2022-01",
    )

    verified = (
        result
        .filter(pl.col("verified_purchase") == True)
        .row(0, named=True)
    )

    non_verified = (
        result
        .filter(pl.col("verified_purchase") == False)
        .row(0, named=True)
    )

    assert verified["review_count"] == 2
    assert verified["avg_review_length"] == pytest.approx(150.0)
    assert verified["avg_rating"] == pytest.approx(4.5)

    assert non_verified["review_count"] == 1
    assert non_verified["avg_review_length"] == pytest.approx(300.0)
    assert non_verified["avg_rating"] == pytest.approx(2.0)


def test_monthly_summary_with_no_matching_rows():
    df = sample_reviews()
    df = add_features(df)

    result = get_monthly_summary(
        df.lazy(),
        start_month="2025-01",
        end_month="2025-12",
    )

    assert result.height == 0