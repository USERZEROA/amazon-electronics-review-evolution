import pandas as pd

from analysis import run_sentiment_model


def test_sentiment_model_training_and_prediction():
    df = pd.DataFrame({
        "text": [
            "good great product",
            "good excellent product",
            "good love this",
            "good perfect item",
            "good amazing quality",
            "good easy to use",
            "bad terrible product",
            "bad awful product",
            "bad poor quality",
            "bad waste of money",
            "bad worst item",
            "bad useless product",
        ],
        "is_positive": [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    })

    result = run_sentiment_model(
        df,
        test_size=0.33,
        random_state=42,
        max_features=100,
        min_df=1,
    )

    assert len(result["predictions"]) == len(
        result["y_test"]
    )

    assert set(
        result["predictions"]
    ).issubset({
        0,
        1,
    })

    assert set(
        result["model"].classes_
    ) == {
        0,
        1,
    }

    assert (
        len(
            result["vectorizer"]
            .get_feature_names_out()
        )
        > 0
    )

    assert 0.0 <= result["accuracy"] <= 1.0

    assert result["accuracy"] >= 0.75