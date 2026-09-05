"""Tests for synthetic data helpers."""

from anomaly_lite.data import make_contaminated_gaussian


def test_make_contaminated_gaussian_shapes_and_labels():
    X, y = make_contaminated_gaussian(
        n_normal=90, n_anomaly=10, n_features=4, random_state=1
    )
    assert X.shape == (100, 4)
    assert y.sum() == 10
    assert set(y.tolist()) == {0, 1}


def test_reproducible_with_seed():
    X1, y1 = make_contaminated_gaussian(random_state=99)
    X2, y2 = make_contaminated_gaussian(random_state=99)
    assert (X1.values == X2.values).all()
    assert (y1 == y2).all()
