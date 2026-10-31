"""
ml_workflow.py — Fonctions communes du projet UE Machine Learning

Étapes couvertes pour l'instant:
    1. Importer les données
    2. Nettoyer et prétraiter (valeurs manquantes, doublons, centrage-réduction, ACP)
    3. Séparer en train / test

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Import des données
# ---------------------------------------------------------------------------
def read_spambase_names(names_path):
    """Lit les 57 noms de features dans le fichier spambase.names."""
    names = []
    with open(names_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(("word_freq_", "char_freq_", "capital_run_length_")):
                names.append(line.split(":")[0].strip())
    return names


def load_spambase(data_dir=DATA_DIR):
    """Charge Spambase. Retourne (X, y) : DataFrame de 57 colonnes et Series 'spam'."""
    data_dir = Path(data_dir)
    names = read_spambase_names(data_dir /"spambase"/"spambase.names")
    df = pd.read_csv(data_dir /"spambase"/"spambase.data", header=None, names=names + ["spam"])
    return df.drop(columns="spam"), df["spam"].astype(int)


def load_diabetes(data_dir=DATA_DIR,
                  filename="diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Charge le dataset Diabetes (BRFSS 2015). Retourne (X, y), y = 'Diabetes_binary'."""
    df = pd.read_csv(Path(data_dir) / filename)
    return df.drop(columns="Diabetes_binary"), df["Diabetes_binary"].astype(int)


def load_dataset(name, data_dir=DATA_DIR):
    """Point d'entrée unique : name ∈ {'spam', 'diabetes'}."""
    loaders = {"spam": load_spambase, "diabetes": load_diabetes}
    if name not in loaders:
        raise ValueError(f"Dataset inconnu : {name!r}. Choix : {list(loaders)}")
    return loaders[name](data_dir)


def describe_dataset(X, y):
    """résumé rapide du dataset : nombre d'échantillons, de features, de valeurs manquantes,"""
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_missing": int(X.isna().sum().sum()), # donne un tableau de booléens ; le premier .sum() compte par colonne, le second fait le total
        "n_duplicates": int(pd.concat([X, y], axis=1).duplicated().sum()), #on colle X et y côte à côte car un doublon, c’est la même ligne et le même label ;
        "class_ratio": y.value_counts(normalize=True).round(3).to_dict(), # donne des proportions
    }


# ---------------------------------------------------------------------------
# 2. Nettoyage / prétraitement
# ---------------------------------------------------------------------------
def remove_duplicates(X, y):
    """Supprime les lignes (X, y) strictement identiques.

    À faire AVANT le split : sinon une même ligne peut se retrouver à la fois
    dans le train et dans le test, ce qui gonfle artificiellement les scores.
    """
    mask = ~pd.concat([X, y], axis=1).duplicated()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def log_transform(X, columns):
    """Difficulté supplémentaire détectée : certaines features (capital_run_length_*
    de Spambase) ont une distribution à queue lourde, avec quelques valeurs extrêmes
    (jusqu'à 15 841 pour une médiane de 95), donc on change l'échelle de ces
    colonnes : x -> log(1 + x).

    Utilité : sans ça, ces quelques valeurs extrêmes dominent les calculs des modèles
    sensibles à l'échelle (régression logistique, k-NN, SVM) et tirent la frontière
    de décision vers elles, même après standardisation (z jusqu'à 25).

    Les valeurs extrêmes restent les plus grandes mais pèsent moins.
    Aucune information perdue, pas de fuite de données (rien n'est appris).
    log1p(x) = log(1 + x) est défini en x = 0, contrairement à log(x).
    """
    X = X.copy()
    X[columns] = np.log1p(X[columns])
    return X


def split_data(X, y, test_size=0.2, random_state=42):
    """3. Séparation train/test stratifiée (garde la même proportion de classes)."""
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


def preprocess(X_train, X_test, impute_strategy="mean", scale=True,
               pca_components=None, random_state=42):
    """Imputation -> centrage-réduction -> ACP (optionnelle).

    Chaque transformation est ajustée sur X_train puis appliquée à X_train et X_test.

    - Imputation : valeur manquante remplacée par la moyenne (ou médiane) du train.
    - Centrage-réduction : z = (x - mu_train) / sigma_train.
    - ACP : si pca_components est un entier, on garde ce nombre de composantes ;
      si c'est un float dans ]0, 1[, on garde le nombre minimal de composantes
      expliquant cette part de variance (ex. 0.9).

    Retourne (X_train_t, X_test_t, transformers) où transformers est un dict des
    objets ajustés (pour les réutiliser ou inspecter l'ACP).
    """
    transformers = {}
    columns = list(X_train.columns)

    imputer = SimpleImputer(strategy=impute_strategy)
    Xtr = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)
    transformers["imputer"] = imputer

    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        transformers["scaler"] = scaler

    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        transformers["pca"] = pca
        columns = [f"PC{i + 1}" for i in range(Xtr.shape[1])]

    Xtr = pd.DataFrame(Xtr, columns=columns, index=X_train.index)
    Xte = pd.DataFrame(Xte, columns=columns, index=X_test.index)
    return Xtr, Xte, transformers


# ---------------------------------------------------------------------------
# Enchaînement des étapes 1 -> 3
# ---------------------------------------------------------------------------

def prepare_data(name, data_dir=DATA_DIR, drop_duplicates=True, log_cols=None,
                 test_size=0.2, impute_strategy="mean", scale=True,
                 pca_components=None, random_state=42):
    """Pipeline complet import -> nettoyage -> split -> prétraitement. (enchainement des fonctions précédentes dans l'ordre)

    Exemple :
        Xtr, Xte, ytr, yte, tr = prepare_data("spam", log_cols=SPAM_HEAVY_TAILED)
        Xtr, Xte, ytr, yte, tr = prepare_data("diabetes", pca_components=0.9)
    """
    X, y = load_dataset(name, data_dir)
    if drop_duplicates:
        X, y = remove_duplicates(X, y)
    if log_cols:
        X = log_transform(X, log_cols)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train, X_test, transformers = preprocess(
        X_train, X_test, impute_strategy, scale, pca_components, random_state)
    return X_train, X_test, y_train, y_test, transformers