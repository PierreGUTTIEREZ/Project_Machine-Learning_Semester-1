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
| **Spambase** | UCI | spam (1) / non-spam (0) | 4 601 e-mails × 57 features continues | 39% de spams |
| **Diabetes Health Indicators (BRFSS 2015)** | Kaggle | diabétique ou prédiabétique (1) / non (0) | 253 680 personnes × 21 features | 14 % de positifs (déséquilibré)|

### Workflow

1. **Import** des données
2. **Nettoyage et prétraitement** : valeurs manquantes, doublons, centrage-réduction, ACP
3. **Séparation** train / test, stratifiée
4. **Entraînement** avec sélection de features
5. **Validation** et comparaison des modèles

Les transformations qui apprennent des paramètres (imputation, standardisation, ACP) sont ajustées **sur le train uniquement**, puis appliquées au test.

## Structure du dépôt

```
.
├── Diabete/                   # dataset Diabetes (diabetes_binary_health_indicators_BRFSS2015.csv)
├── spambase/                  # dataset Spambase (spambase.data, spambase.names, spambase.DOCUMENTATION)
├── ml_workflow.py             # toutes les fonctions du workflow
├── test_ml_workflow.py        # tests unitaires
├── spambase.ipynb             # analyse + résultats Spambase
├── diabetes.ipynb             # analyse + résultats Diabetes
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <url-du-depot>
cd Project_Machine-Learning_Semester-1
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate (mettre ça avant si ça ne marche pas Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

Bibliothèques nécessaires pour l'exécution : numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, pytest, torch

```bash
pip install -r requirements.txt
```
ou 
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter pytest torch
```

## Comment lancer les tests unitaires

Tests :

```bash
pytest -q
```
16/16 passed : tous les tests unitaires sont validés.

## Conventions de travail en groupe

- Le code utilisable est dans `ml_workflow.py` ; les notebooks se contentent de l'appeler et de commenter les résultats.
- Répartition des tâches de façon équitable.
- Création d'un groupe de discussion pour s'organiser et discuter de l'avancée du travail.

## Utilisation de l'IA
L'IA a été utilisée pour s'informer sur l'ensemble des méthodes possibles et sur les choix les plus pertinents pour ce sujet. Elle a aussi été utilisée pour comprendre la syntaxe des tests unitaires ainsi qu'une partie de leur rédaction.
Avant le rendu final, nous lui avons finalement demander de vérifier la cohérence des deux notebooks et s'il y avait des oublis.
