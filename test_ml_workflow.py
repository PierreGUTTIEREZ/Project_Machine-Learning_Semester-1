from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import ml_workflow as mw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy():
    """Jeu de données synthétique pour les tests rapides."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 3, size=(100, 4)), columns=list("abcd"))
    X.iloc[0, 0] = np.nan 
    y = pd.Series([0] * 70 + [1] * 30)
    return X, y


@pytest.fixture
def clean_toy():
    """Jeu synthétique sans NaN, parfaitement séparable/apprenable."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series([0] * 30 + [1] * 30)
    return X, y


# ---------------------------------------------------------------------------
# 1. Tests d'import et de description des données
# ---------------------------------------------------------------------------

def test_load_spambase_shape():
    """Vérifie la forme et les classes du dataset Spambase."""
    data_path = mw.DATA_DIR / "spambase" / "spambase.data"
    if not data_path.exists():
        pytest.skip(f"Fichier absent : {data_path}")
    X, y = mw.load_spambase()
    assert X.shape == (4601, 57)
    assert set(y.unique()) == {0, 1}


def test_load_diabetes_shape():
    """Vérifie la forme et les classes du dataset Diabetes."""
    csv_path = mw.DATA_DIR / "Diabete" / "diabetes_binary_health_indicators_BRFSS2015.csv"
    if not csv_path.exists():
        pytest.skip(f"Fichier absent : {csv_path}")
    X, y = mw.load_diabetes()
    assert X.shape == (253680, 21)
    assert set(y.unique()) == {0, 1}


def test_load_dataset_invalid_name():
    """Vérifie qu'un nom de dataset inconnu lève une ValueError."""
    with pytest.raises(ValueError, match="Dataset inconnu"):
        mw.load_dataset("dataset_inexistant")


def test_describe_dataset(toy):
    """Vérifie les statistiques renvoyées par describe_dataset."""
    X, y = toy
    desc = mw.describe_dataset(X, y)
    assert desc["n_samples"] == 100
    assert desc["n_features"] == 4
    assert desc["n_missing"] == 1
    assert desc["n_duplicates"] == 0
    assert desc["class_ratio"] == {0: 0.7, 1: 0.3}


# ---------------------------------------------------------------------------
# 2. Tests de nettoyage et prétraitement
# ---------------------------------------------------------------------------

def test_remove_duplicates():
    """Vérifie la détection et la suppression des doublons stricts (X, y)."""
    X = pd.DataFrame({"a": [1, 1, 2]})
    y = pd.Series([0, 0, 0])
    Xd, yd = mw.remove_duplicates(X, y)
    assert len(Xd) == len(yd) == 2


def test_log_transform():
    """Vérifie que log1p est appliqué aux colonnes cibles uniquement."""
    X = pd.DataFrame({"a": [0.0, 1.0, 9.0], "b": [10.0, 20.0, 30.0]})
    X_log = mw.log_transform(X, columns=["a"])
    assert X_log["a"].iloc[0] == pytest.approx(0.0)
    assert X_log["a"].iloc[1] == pytest.approx(np.log(2))
    assert X_log["b"].tolist() == [10.0, 20.0, 30.0]


def test_split_is_stratified(toy):
    """Vérifie que le split train/test respecte la stratification des classes."""
    X, y = toy
    _, _, ytr, yte = mw.split_data(X, y, test_size=0.2, random_state=42)
    assert len(yte) == 20
    assert ytr.mean() == pytest.approx(0.3)
    assert yte.mean() == pytest.approx(0.3)


def test_preprocess_fits_on_train_only(toy):
    """Vérifie l'imputation et le centrage-réduction calculés sur le train uniquement."""
    X, y = toy
    Xtr, Xte, _, _ = mw.split_data(X, y, random_state=42)
    Xtr_t, Xte_t, transformers = mw.preprocess(Xtr, Xte)
    assert not Xtr_t.isna().any().any()
    assert not Xte_t.isna().any().any()
    np.testing.assert_allclose(Xtr_t.mean(), 0, atol=1e-10)
    np.testing.assert_allclose(Xtr_t.std(ddof=0), 1, atol=1e-10)
    assert "imputer" in transformers and "scaler" in transformers


def test_pca_variance_threshold(toy):
    """Vérifie que l'ACP conserve le seuil de variance demandée."""
    X, y = toy
    Xtr, Xte, _, _ = mw.split_data(X, y, random_state=42)
    Xtr_t, Xte_t, tr = mw.preprocess(Xtr, Xte, pca_components=0.9, random_state=42)
    assert tr["pca"].explained_variance_ratio_.sum() >= 0.9
    assert Xtr_t.shape[1] == Xte_t.shape[1] <= 4


# ---------------------------------------------------------------------------
# 3. Tests de sélection de features et hyperparamètres
# ---------------------------------------------------------------------------

def test_choose_k_knn(clean_toy):
    """Vérifie la sélection de k pour le KNN par cross-validation."""
    X, y = clean_toy
    best_k, ks, scores = mw.choose_k_knn(X, y, k_values=[1, 3, 5])
    assert best_k in [1, 3, 5]
    assert len(scores) == 3


def test_with_feature_selection(clean_toy):
    """Vérifie le pipeline avec SelectKBest."""
    X, y = clean_toy
    pipe = mw.with_feature_selection(LogisticRegression(), n_features=2)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_choose_n_features(clean_toy):
    """Vérifie la sélection du nombre optimal de features."""
    X, y = clean_toy
    model = LogisticRegression()
    best_n, n_vals, scores = mw.choose_n_features(model, X, y, n_values=[2, 3, 4])
    assert best_n in [2, 3, 4]
    assert len(scores) == len(n_vals)


# ---------------------------------------------------------------------------
# 4. Tests Réseau de Neurones PyTorch & Scikit-learn wrapper
# ---------------------------------------------------------------------------

def test_torch_mlp_fit_predict(clean_toy):
    """Vérifie l'entraînement et la prédiction de la classe TorchMLP."""
    X, y = clean_toy
    mlp = mw.TorchMLP(hidden_size1=8, hidden_size2=4, epochs=10, lr=0.1)
    mlp.fit(X, y)
    preds = mlp.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# 5. Tests d'évaluation et de métriques
# ---------------------------------------------------------------------------

def test_get_models():
    """Vérifie le dictionnaire de modèles retourné par get_models."""
    models = mw.get_models(k_knn=5, class_weight=None)
    expected_keys = {"KNN", "Régression logistique", "Random Forest", "Réseau de neurones"}
    assert set(models.keys()) == expected_keys


def test_evaluate_on_test(clean_toy):
    """Vérifie les métriques calculées par evaluate_on_test."""
    X, y = clean_toy
    model = LogisticRegression()
    model.fit(X, y)
    results = mw.evaluate_on_test(model, X, y)

    expected_metrics = {"accuracy", "précision", "rappel", "faux positifs", "faux négatifs"}
    assert expected_metrics.issubset(results.keys())
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["précision"] <= 1.0
    assert 0.0 <= results["rappel"] <= 1.0
    assert results["faux positifs"] >= 0
    assert results["faux négatifs"] >= 0


def test_add_fbeta():
    """Vérifie le calcul correct du F-beta."""
    df = pd.DataFrame([{"précision": 0.8, "rappel": 0.5}])
    df = mw.add_fbeta(df, beta=2)
    expected_f2 = 5 * (0.8 * 0.5) / (4 * 0.8 + 0.5)
    assert df["F2"].iloc[0] == pytest.approx(expected_f2, rel=1e-3)
