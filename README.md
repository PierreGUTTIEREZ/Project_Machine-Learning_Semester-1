# Project_Machine-Learning_Semester-1

Projet de l'UE A *Machine Learning*, TAF MCE, IMT Atlantique

## Membres du projet:
Inès Oualdi-Djebril
Pierre Guttierez
Guillaume Leroy
Agathe Coignard

## Objectif

1. Mettre en place un **workflow de Machine Learning réutilisable** et l'appliquer à deux problèmes de **classification binaire** (Spambase et Diabetes)
2. Entraîner et comparer plusieurs méthodes, puis **justifier le choix final** pour chaque dataset à l'aide de métriques adaptées

| Dataset | Source | Tâche | Taille | Particularités |
|---|---|---|---|---|
| **Spambase** | UCI | spam (1) / non-spam (0) | 4 601 e-mails × 57 features continues | 39 % de spam, 0 valeur manquante, 391 doublons, features à queue lourde (`capital_run_length_*`) ; les faux positifs coûtent cher |
| **Diabetes Health Indicators (BRFSS 2015)** | Kaggle | diabétique ou prédiabétique (1) / non (0) | 253 680 personnes × 21 features | 14 % de positifs (déséquilibré), features surtout binaires/ordinales, 24 206 doublons |

### Workflow

1. **Import** des données
2. **Nettoyage et prétraitement** : valeurs manquantes, doublons, centrage-réduction, ACP
3. **Séparation** train / test, stratifiée
4. **Entraînement** avec sélection de features
5. **Validation** et comparaison des modèles

Les transformations qui apprennent des paramètres (imputation, standardisation, ACP) sont ajustées **sur le train uniquement**, puis appliquées au test. La standardisation fait :

$$
z_{ij} = \frac{x_{ij} - \mu_j^{\text{train}}}{\sigma_j^{\text{train}}}
$$

## Structure du dépôt

```
.
├── data/                      # datasets (spambase.*, diabetes_*.csv)
├── ml_workflow.py             # toutes les fonctions du workflow
├── test_ml_workflow.py        # tests unitaires
├── spam.ipynb                 # analyse + résultats Spambase
├── diabetes.ipynb             # analyse + résultats Diabetes
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <url-du-depot>
cd Project_Machine-Learning_Semester-1
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Le CSV Diabetes est à télécharger sur Kaggle et à placer dans `data/` s'il n'est pas versionné.

Bibliothèques nécessaires pour l'exécution : numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, pytest

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter pytest
```

## Utilisation: comment appeler les fonctions du workflow depuis un notebook

```python
import ml_workflow as mw

# Spambase : log(1+x) sur les features à queue lourde, puis standardisation
X_train, X_test, y_train, y_test, tr = mw.prepare_data(
    "spam", log_cols=mw.SPAM_HEAVY_TAILED)

# Diabetes avec ACP gardant 90 % de la variance
X_train, X_test, y_train, y_test, tr = mw.prepare_data(
    "diabetes", pca_components=0.9)
```

Tests :

```bash
pytest -q
```

## Conventions de travail en groupe

- Une branche par fonctionnalité (`feature/feature-selection`, `feature/models-spam`…), fusionnée dans `main` via *merge request*.
- Le code réutilisable va dans `ml_workflow.py` ; les notebooks se contentent de l'appeler et de commenter les résultats.
- Pas de sorties lourdes dans les commits de notebooks : faire *Restart & Run All* avant de commit.

## Utilisation de l'IA
L'IA a été utilisée dans le cadre de l'information sur l'ensemble des méthodes possibiles de modèles de classification binaire et sur les choix les plus pertinents pour ce sujet.
