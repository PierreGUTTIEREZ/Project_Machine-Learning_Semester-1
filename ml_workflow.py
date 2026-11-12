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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


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
    df = pd.read_csv(Path(data_dir) / "Diabete" / filename)
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

# ---------------------------------------------------------------------------
# 4. Entraînement : réglage des modèles et sélection de features
# ---------------------------------------------------------------------------

def choose_k_knn(X_train, y_train, k_values):
    """Choisit le nombre de voisins k par validation croisée (5 folds) sur le train."""
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(knn, X_train, y_train).mean())
    best_k = k_values[np.argmax(scores)]
    return best_k, k_values, scores


def with_feature_selection(model, n_features):
    """Pipeline : on garde les n_features meilleures features (test ANOVA), puis le modèle.
    La sélection est refaite à chaque fold de la validation croisée, donc pas de fuite."""
    return make_pipeline(SelectKBest(f_classif, k=n_features), model)


def choose_n_features(model, X_train, y_train, n_values):
    """Choisit le nombre de features à garder par validation croisée sur le train."""
    scores = []
    for n in n_values:
        pipe = with_feature_selection(model, n)
        scores.append(cross_val_score(pipe, X_train, y_train).mean())
    best_n = n_values[np.argmax(scores)]
    return best_n, n_values, scores


# ---------------------------------------------------------------------------
# 5. Réseau de neurones
# ---------------------------------------------------------------------------

class mlp_classification_nn(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc0 = nn.Linear(input_size, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc0(x))
        out = self.relu(self.fc1(out))
        y_pred = self.sigmoid(self.fc2(out))
        return y_pred


def to_tensor(X):
    """DataFrame / Series / array numpy -> tenseur PyTorch en float32."""
    return torch.tensor(np.array(X)).float()


def train_mlp(X_train, y_train, hidden_size1, hidden_size2, epochs, lr,
              X_val=None, y_val=None):
    """Entraîne le MLP (descente de gradient sur tout le train à chaque epoch).
    Si on donne X_val et y_val, on garde aussi la loss de validation à chaque epoch."""
    torch.manual_seed(42)
    x = to_tensor(X_train)
    y = to_tensor(y_train).reshape(-1, 1)
    if X_val is not None:
        x_val = to_tensor(X_val)
        y_val = to_tensor(y_val).reshape(-1, 1)

    model = mlp_classification_nn(x.shape[1], hidden_size1, hidden_size2, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    val_losses = []
    for epoch in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if X_val is not None:
            val_loss = criterion(model(x_val), y_val)
            val_losses.append(val_loss.item())

    return model, losses, val_losses


def predict_mlp(model, X):
    """Classe prédite : 1 (spam) si la probabilité en sortie dépasse 0.5, sinon 0."""
    proba = model(to_tensor(X)).detach().numpy().ravel()
    return (proba > 0.5).astype(int)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Pour utiliser notre MLP PyTorch comme un modèle scikit-learn (fit / predict),
    et donc le passer à cross_val_score, choose_n_features, compare_models..."""

    def __init__(self, hidden_size1, hidden_size2, epochs, lr):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # attendu par scikit-learn pour un classifieur
        self.model_, self.losses_, _ = train_mlp(X, y, self.hidden_size1, self.hidden_size2,
                                                 self.epochs, self.lr)
        return self

    def predict(self, X):
        return predict_mlp(self.model_, X)


# ---------------------------------------------------------------------------
# 6. Validation et comparaison
# ---------------------------------------------------------------------------

def get_models(k_knn, class_weight):
    """Les 4 modèles comparés.
    class_weight = "balanced" si le dataset est déséquilibré (Diabetes), None sinon.
    Seuls la régression logistique et le Random Forest acceptent ce paramètre."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=k_knn),
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight=class_weight),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight=class_weight,
                                                random_state=42),
        "Réseau de neurones": TorchMLP(32, 16, 1500, 0.1),
    }


def compare_models(models, X_train, y_train):
    """Compare les modèles par validation croisée (5 folds) sur le train.
    Le test n'est pas utilisé ici."""
    results = {}
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train,
                             scoring=["accuracy", "precision", "recall"])
        results[name] = {"accuracy": res["test_accuracy"].mean(),
                         "précision": res["test_precision"].mean(),
                         "rappel": res["test_recall"].mean()}
    return pd.DataFrame(results).T


def evaluate_on_test(model, X_test, y_test):
    """Scores sur le test d'un modèle déjà entraîné (à faire une seule fois, à la fin)."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  # [[VN, FP], [FN, VP]]
    return {"accuracy": accuracy_score(y_test, y_pred),
            "précision": precision_score(y_test, y_pred),
            "rappel": recall_score(y_test, y_pred),
            "faux positifs": cm[0, 1],
            "faux négatifs": cm[1, 0]}

def add_fbeta(df, beta=2):
    """Ajoute une colonne F-beta, calculée à partir des colonnes 'précision' et 'rappel'.
    beta > 1 : le rappel compte plus ;
    beta < 1 : la précision compte plus"""
    
    prec, rap = df["précision"], df["rappel"]
    df[f"F{beta}"] = (1 + beta**2) * prec * rap / (beta**2 * prec + rap)
    return df