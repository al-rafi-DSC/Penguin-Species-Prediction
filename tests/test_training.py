# This pytest file for testing machine learning training pipeline.

from train import FEATURES, build_pipeline, load_data


def test_load_data_returns_expected_features() -> None:
    features, target = load_data()

    assert list(features.columns) == FEATURES
    assert len(features) == len(target)
    assert len(features) == 344
    assert target.nunique() == 3


def test_pipeline_can_train_and_predict() -> None:
    features, target = load_data()
    pipeline = build_pipeline()

    pipeline.fit(features, target)
    predictions = pipeline.predict(features.head(3))

    assert len(predictions) == 3
    assert set(predictions).issubset({"Adelie", "Chinstrap", "Gentoo"})
