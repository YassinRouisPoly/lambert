# Lambert

**Auteur** : M. Rouis  
**Encadrant** : Pr. Salah GONTARA  
**Nature du projet** : Projet académique MLOps

## Introduction

Lambert est un modèle de langage basé sur **CamemBERT**, conçu pour assister la rédaction de textes juridiques en langue française.

Le modèle est **intégralement entraîné sur des données juridiques françaises**, ce qui le rend particulièrement adapté aux usages liés au droit français (rédaction, reformulation, aide à l’analyse de textes légaux).

Ce projet s’inscrit dans un **cadre académique MLOps**, avec un accent mis sur la reproductibilité, la traçabilité des expériences et l’industrialisation du cycle de vie des modèles de machine learning.

## Dataset

&#x20;

Le jeu de données principal utilisé pour l’entraînement est **COLD (Corpus Of Legal Documents)**, un dataset juridique spécialisé regroupant des articles de lois et documents légaux français.

## Prérequis

- Python 3.12.x
- Docker
- DVC

## Technologies utilisées

- **MinIO** : stockage objet compatible S3 pour les datasets et artefacts de modèles.
- **FastAPI** : exposition du modèle via une API REST performante et typée.
- **MLflow** : suivi des expérimentations, métriques et modèles.
- **Optuna** : optimisation automatique des hyperparamètres.
- **ZenML** : orchestration et reproductibilité des pipelines MLOps.

## Structure du projet

```
.
├── .dvc/                 # Configuration DVC
├── .github/
│   └── workflows/        # Pipelines CI/CD (GitHub Actions)
├── cache/                # Cache intermédiaire (prétraitement, artefacts temporaires)
├── client/               # Interface cliente
├── datasets/
│   └── original/         # Dataset source (COLD – French Law)
├── libs/                 # Utilitaires et briques communes
├── models/               # Modèles entraînés et artefacts associés
├── server/               # API backend
├── .dvcignore
├── .gitignore
├── datasets.dvc          # Suivi DVC des datasets
├── models.dvc            # Suivi DVC des modèles
├── docker-compose.yml    # Orchestration des services
├── mlflow.db             # Backend MLflow (tracking local)
├── optimized_train.py    # Pipeline ZenML d'entraînement avec optimisation des hyperparamètres (Optuna)
├── pipeline.py           # Pipeline ML (ZenML)
├── prepare.sh            # Initialisation du projet
├── run.sh                # Lancement global
├── requirements.txt
└── README.md
```

La gestion des versions de datasets et de modèles est assurée par **DVC** ; seules les sources originales sont visibles dans l’arborescence.

## Aperçu des pipelines MLOps

Cette section présente une vue synthétique des pipelines **ZenML** et du suivi des expérimentations via **MLflow**.

### Pipeline ZenML – Data Cleaning

Ce pipeline est responsable de l’ingestion et du nettoyage des données juridiques avant toute phase d’entraînement.

### Pipeline ZenML – Training

Ce pipeline orchestre l’entraînement du modèle, l’optimisation des hyperparamètres et la validation, avec traçabilité complète des runs.

### Suivi des expérimentations – MLflow

Les différentes expériences, métriques et artefacts générés par les pipelines sont centralisés et comparables via **MLflow**.

## Préparation

Pour préparer la structure du projet et générer les fichiers manquants (téléchargement du dataset original, initialisation des dossiers, etc.), exécutez le script suivant :

```sh
sh ./prepare.sh
```

## Démarrage

### Installation des dépendances

```sh
pip install -r requirements.txt
```

### Lancement du projet

L’ensemble du projet peut être démarré via le script dédié :

```sh
sh ./run.sh
```

Ce script orchestre le démarrage des services nécessaires (environnement, backend et composants associés).

## Exécution des pipelines ZenML

Chaque pipeline MLOps peut être exécuté indépendamment.

Les pipelines standard (data cleaning, entraînement) sont définis et exécutables via le fichier :

```sh
pipeline.py
```