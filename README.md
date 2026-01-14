# Lambert

## Introduction
Lambert est un modèle de langage basé sur **CamemBERT**, conçu pour assister la rédaction de textes juridiques en langue française.

Le modèle est **intégralement entraîné sur des données juridiques françaises**, ce qui le rend particulièrement adapté aux usages liés au droit français (rédaction, reformulation, aide à l’analyse de textes légaux).

## Dataset
<img src="https://lil-blog-media.s3.amazonaws.com/COLDfrenchlaw.webp" alt="COLD French Law" width="600"/><br/>
<img src="https://cdn-thumbnails.huggingface.co/social-thumbnails/datasets/harvard-lil/cold-french-law.png" alt="COLD French Law Dataset" width="600"/><br/>

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

<img src="https://i.imgur.com/xkZ5KSp.png" alt="ZenML Data Cleaning Pipeline" width="800"/>

### Pipeline ZenML – Training

Ce pipeline orchestre l’entraînement du modèle, l’optimisation des hyperparamètres et la validation, avec traçabilité complète des runs.

<img src="https://i.imgur.com/0X6FDs8.png" alt="ZenML Training Pipeline" width="800"/>

### Suivi des expérimentations – MLflow

Les différentes expériences, métriques et artefacts générés par les pipelines sont centralisés et comparables via **MLflow**.

<img src="https://i.imgur.com/ZzQ2jDc.png" alt="MLflow Experiments" width="800"/>

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

Ce point d’entrée permet de lancer les pipelines ZenML sans démarrer l’ensemble de l’infrastructure applicative.


## Endpoint d’inférence

Une API d’inférence est exposée via **FastAPI**, permettant d’obtenir des prédictions à partir d’un texte juridique en entrée.

### Endpoint

- **Méthode** : `POST`
- **Route** : `/get_prediction`
- **Données (JSON)** :
    - `text` : Texte à traiter

```http
# Exemple de requête
POST /get_prediction HTTP/1.1
Host: localhost
Content-Type: application/json

{ "text": "La république fran" }

```
```http
# Exemple de réponse
HTTP/1.1 200 OK
Date: ...
Server: ...
Last-Modified: ...
Content-Length: ...
Content-Type: application/json
Connection: Closed

{
    "preferredPrediction": "this",
    "predictions": {
        "this": [ "est", "française", "dispose", ...],
        "next": [ "est", "française", "dispose", ...],
        "pendingNext": [ "est", "française", "dispose", ...]
    },
    "updates": {
        "this": [ "est", "française", "dispose", ...],
        "next": "La république française est",
        "pendingNext": "La république française est"
    }
}
```

Cet endpoint reçoit un texte et retourne :
- la prédiction privilégiée du modèle,
- les prédictions pour l’état courant et suivant,
- les mises à jour associées.

Cet endpoint permet l’intégration du modèle dans des applications clientes ou des services tiers, en mode **inférence temps réel**.
