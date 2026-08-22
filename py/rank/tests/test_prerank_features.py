"""Pure feature-vector construction and scaling, tested against hand-computed
values -- matching this repo's style for pure retrieval-math functions."""

from __future__ import annotations

from rank.prerank_features import FeatureScaler, build_feature_vector, fit_scaler


def test_build_feature_vector_is_a_plain_tuple():
    result = build_feature_vector(bm25_score=12.5, dense_score=0.3, doc_length=200)
    assert result == (12.5, 0.3, 200.0)


def test_fit_scaler_hand_computed_mean_and_std():
    vectors = [(0.0, 0.0, 0.0), (10.0, 2.0, 100.0)]
    scaler = fit_scaler(vectors)
    assert scaler.mean == (5.0, 1.0, 50.0)
    # population std (ddof=0): sqrt(((0-5)^2 + (10-5)^2) / 2) = 5.0
    assert scaler.std == (5.0, 1.0, 50.0)


def test_scaler_transform_zscore():
    scaler = FeatureScaler(mean=(5.0, 1.0, 50.0), std=(5.0, 1.0, 50.0))
    result = scaler.transform((10.0, 2.0, 100.0))
    assert result == (1.0, 1.0, 1.0)


def test_scaler_transform_guards_zero_std():
    scaler = FeatureScaler(mean=(5.0, 1.0, 50.0), std=(0.0, 1.0, 50.0))
    result = scaler.transform((5.0, 2.0, 100.0))
    # first feature's std is 0 -- every value equals the mean, so it
    # normalizes to 0.0 rather than dividing by zero.
    assert result == (0.0, 1.0, 1.0)


def test_fit_scaler_single_vector_has_zero_std():
    scaler = fit_scaler([(3.0, 4.0, 5.0)])
    assert scaler.mean == (3.0, 4.0, 5.0)
    assert scaler.std == (0.0, 0.0, 0.0)
