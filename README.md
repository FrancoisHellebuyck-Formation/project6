# Detection du Cancer du Poumon - Projet MLOps

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![MLflow](https://img.shields.io/badge/MLflow-3.4.0-0194E2)

Projet d'initiation au MLOps visant à développer un système de prédiction du risque de cancer du poumon basé sur des données de santé et de facteurs de risque.

## Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du Projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des Données](#structure-des-données)
- [Notebooks](#notebooks)
- [Modules](#modules)
- [MLflow](#mlflow)
- [Screenshots](#screenshots)
- [Développement](#développement)
- [Contribuer](#contribuer)

## Vue d'ensemble

Ce projet a pour objectif d'aider à prédire le risque de cancer du poumon à faible coût en utilisant des techniques de machine learning. Les données sont collectées à partir d'un système de prédiction en ligne et incluent des informations démographiques ainsi que des facteurs de risque comportementaux et médicaux.

### Objectifs

- Développer un modèle de classification pour la détection du cancer du poumon
- Implémenter un pipeline MLOps complet avec tracking des expérimentations
- Optimiser les performances du modèle via différentes techniques (cross-validation, optimisation de seuil)
- Assurer la reproductibilité et la traçabilité des expérimentations

### Technologies Utilisées

- **Python 3.13+**
- **MLflow** : Tracking des expérimentations et gestion des modèles
- **Scikit-learn** : Algorithmes de machine learning
- **LightGBM & XGBoost** : Modèles de gradient boosting
- **SHAP** : Explainabilité des modèles
- **Pandas & NumPy** : Manipulation de données
- **Matplotlib & Seaborn** : Visualisation

## Architecture du Projet

```
project6/
├── data/
│   ├── raw/                    # Données brutes (CSV)
│   └── processed/              # Données nettoyées et transformées
├── notebooks/
│   ├── 00-preparation.ipynb    # Préparation et nettoyage des données
│   ├── 01-exploration.ipynb    # Analyse exploratoire (EDA)
│   ├── 02-features.ipynb       # Engineering des features
│   ├── 03-Modeling.ipynb       # Modélisation et expérimentation
│   └── utils_mlflow.ipynb      # Utilitaires MLflow
├── src/
│   ├── models/
│   │   ├── modelization.py     # Métriques et évaluation
│   │   └── seuil.py            # Optimisation de seuils
│   ├── visualization/
│   │   ├── visu.py             # Visualisations avancées
│   │   └── visu_text.py        # Affichages texte
│   ├── constantes.py           # Configuration MLflow et tags
│   └── utils_mlflow.py         # Fonctions utilitaires MLflow
├── mlflow/                      # Artefacts MLflow
├── screenshots/                 # Captures d'écran du projet
├── pyproject.toml              # Configuration du projet
└── main.py                     # Point d'entrée principal
```

## Prérequis

- Python >= 3.13
- uv (gestionnaire de paquets Python)
- Git

## Installation

### 1. Cloner le dépôt

```bash
git clone <repository-url>
cd project6
```

### 2. Créer l'environnement virtuel et installer les dépendances

```bash
# Avec uv (recommandé)
uv venv
source .venv/bin/activate  # Sur Linux/Mac
# ou
.venv\Scripts\activate     # Sur Windows

uv pip install -e .
```

### 3. Configuration de MLflow

Le serveur MLflow est configuré pour tourner sur `http://127.0.0.1:5010` par défaut. Pour modifier cette configuration, définir la variable d'environnement :

```bash
export MLFLOW_TRACKING_URI="http://your-mlflow-server:port"
```

## Utilisation

### Démarrage rapide

1. **Lancer le serveur MLflow** :

```bash
mlflow ui --port 5010
```

2. **Exécuter les notebooks dans l'ordre** :
   - `00-preparation.ipynb` : Préparation des données
   - `01-exploration.ipynb` : Exploration et analyse
   - `02-features.ipynb` : Feature engineering
   - `03-Modeling.ipynb` : Entraînement et évaluation des modèles

### Exécution via script

```bash
python main.py
```

## Structure des Données

### Données brutes

Le projet utilise deux fichiers CSV initiaux :

1. **Patient_Lung_Cancer_Dataset.csv** (20 000 lignes, 4 colonnes)
   - ID
   - GENDER (F/M)
   - AGE
   - LUNG_CANCER (NO/YES)

2. **Risks_Factors_Lung_Cancer_Dataset.csv** (20 000 lignes, 14 colonnes)
   - ID
   - SMOKING
   - YELLOW_FINGERS
   - ANXIETY
   - PEER_PRESSURE
   - CHRONIC DISEASE
   - FATIGUE
   - ALLERGY
   - WHEEZING
   - ALCOHOL CONSUMING
   - COUGHING
   - SHORTNESS OF BREATH
   - SWALLOWING DIFFICULTY
   - CHEST PAIN

### Transformations

- Fusion des deux datasets sur l'ID
- Suppression des valeurs manquantes (2 131 lignes) → 17 940 observations finales
- Encodage binaire des variables catégorielles :
  - GENDER: F=0, M=1
  - Autres variables (YES/NO): YES=1, NO=0

### Données traitées

Les données nettoyées sont sauvegardées dans `data/processed/` au format CSV et Parquet.

## Notebooks

### 00-preparation.ipynb
- Chargement et fusion des données brutes
- Nettoyage et traitement des valeurs manquantes
- Encodage des variables catégorielles
- Export des données nettoyées

### 01-exploration.ipynb
- Analyse exploratoire des données (EDA)
- Visualisations statistiques
- Étude des corrélations
- Analyse du déséquilibre des classes

### 02-features.ipynb
- Feature engineering
- Sélection de variables
- Transformation des features
- Export des features finales

### 03-Modeling.ipynb
- Entraînement de modèles de classification
- Comparaison de modèles baseline
- Cross-validation
- Optimisation de seuils
- Optimisation hyperparamètres (LightGBM)
- Analyse SHAP
- Enregistrement des modèles dans MLflow

### utils_mlflow.ipynb
- Gestion des expérimentations MLflow
- Nettoyage des runs
- Visualisations interactives

## Modules

### src/models/

#### modelization.py
Fonctions pour l'évaluation des modèles de classification :
- Calcul de métriques (accuracy, precision, recall, F1, etc.)
- Matrices de confusion
- Courbes ROC et Precision-Recall

#### seuil.py
Optimisation des seuils de décision :
- Recherche du seuil optimal
- Évaluation de différentes métriques selon le seuil
- Gestion du trade-off entre précision et rappel

### src/visualization/

#### visu.py
Visualisations avancées :
- Courbes ROC multi-classes
- Feature importance
- Distributions de probabilités
- Graphiques SHAP

#### visu_text.py
Affichages formatés dans les notebooks :
- Informations sur les DataFrames
- Résumés statistiques
- Tables de résultats

### src/utils_mlflow.py

Utilitaires pour l'intégration MLflow :
- Configuration des expérimentations
- Logging de métriques et paramètres
- Sauvegarde d'artefacts
- Gestion des tags

### src/constantes.py

Configuration centralisée :
- URI de tracking MLflow
- Tags par défaut pour les expérimentations
- Nomenclature des expérimentations
- Métadonnées du projet

## MLflow

### Configuration

Le projet utilise MLflow pour le tracking des expérimentations avec la configuration suivante :

- **Tracking URI** : `http://127.0.0.1:5010`
- **Expérimentation principale** : "Lung Cancer Detection"
- **Tags par défaut** :
  - project: lung_cancer
  - team: DataScience
  - department: Openclassrooms
  - owner: francois.hellebuyck

### Types d'expérimentations

Le projet implémente plusieurs types d'expérimentations :

1. **Baseline** (`TAGS_BASELINE`) : Comparaison de modèles de base
2. **Cross-validation** (`TAGS_CV`) : Validation croisée
3. **Threshold Optimization** (`TAGS_TO`) : Optimisation des seuils
4. **LGBM Optimization** (`TAGS_LGBMO`) : Optimisation hyperparamètres LightGBM
5. **Final Model** (`TAGS_FINAL_MODEL`) : Modèle final retenu

### Lancement du serveur MLflow

```bash
mlflow ui --port 5010
```

Accéder à l'interface : [http://127.0.0.1:5010](http://127.0.0.1:5010)

## Screenshots

Le dossier [screenshots/](screenshots/) contient des captures d'écran illustrant les différentes étapes et résultats du projet. 
Ces captures d'écran permettent de visualiser rapidement les résultats clés du projet et le fonctionnement du pipeline MLOps.

## Développement

### Linter et formatage

Le projet utilise Flake8 pour le linting avec les configurations suivantes :

```bash
# Vérifier le code
flake8 src/

# Configuration dans pyproject.toml
# - max-line-length: 88
# - Ignore: E203, W503, E402, E501
```

### Structure de versioning

Le projet suit les bonnes pratiques Git :
- Branche principale : `develop`
- Commits récents trackés dans l'historique

### Ajout de nouvelles fonctionnalités

1. Créer une nouvelle branche
2. Développer la fonctionnalité
3. Ajouter des tests si applicable
4. Mettre à jour la documentation
5. Créer une pull request

## Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche pour votre feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est développé dans le cadre d'une formation OpenClassrooms.

## Contact

**Propriétaire** : François Hellebuyck
**Équipe** : DataScience
**Organisation** : OpenClassrooms

---

*Projet réalisé dans le cadre du parcours "Initiez-vous au MLOps" - OpenClassrooms*
