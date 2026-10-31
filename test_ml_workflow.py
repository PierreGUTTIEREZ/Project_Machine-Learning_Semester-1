"""
Tests unitaires
Lancer avec : pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import ml_workflow as mw


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 3, size=(100, 4)), columns=list("abcd"))
    X.iloc[0, 0] = np.nan
    y = pd.Series([0] * 70 + [1] * 30)
    return X, y


def test_split_is_stratified(toy):
    X, y = toy
    _, _, ytr, yte = mw.split_data(X, y, test_size=0.2)
    assert len(yte) == 20
    assert ytr.mean() == pytest.approx(0.3) and yte.mean() == pytest.approx(0.3)


def test_preprocess_fits_on_train_only(toy):
    X, y = toy
    Xtr, Xte, _, _ = mw.split_data(X, y)
    Xtr_t, Xte_t, _ = mw.preprocess(Xtr, Xte)
    assert not Xtr_t.isna().any().any() and not Xte_t.isna().any().any()
    # Le train est centré-réduit exactement, le test seulement approximativement
    np.testing.assert_allclose(Xtr_t.mean(), 0, atol=1e-10)
    np.testing.assert_allclose(Xtr_t.std(ddof=0), 1, atol=1e-10)


def test_pca_variance_threshold(toy):
    X, y = toy
    Xtr, Xte, _, _ = mw.split_data(X, y)
    Xtr_t, Xte_t, tr = mw.preprocess(Xtr, Xte, pca_components=0.9)
    assert tr["pca"].explained_variance_ratio_.sum() >= 0.9
    assert Xtr_t.shape[1] == Xte_t.shape[1] <= 4


def test_remove_duplicates():
    X = pd.DataFrame({"a": [1, 1, 2]})
    y = pd.Series([0, 0, 0])
    Xd, yd = mw.remove_duplicates(X, y)
    assert len(Xd) == len(yd) == 2


def test_load_spambase_shape():
    if not (mw.DATA_DIR / "spambase.data").exists():
        pytest.skip("données absentes")
    X, y = mw.load_spambase()
    assert X.shape == (4601, 57) and set(y.unique()) == {0, 1}
