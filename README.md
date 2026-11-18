# Panier Vert - MSPR Data Mining

## Contexte

Panier Vert est un réseau fictif de 42 épiceries bio disposant d'un service de commande en ligne et de retrait en magasin.

La direction commerciale souhaite exploiter les historiques de commandes et les données publiques produits afin de :

- fidéliser les clients ;
- améliorer les recommandations ;
- suivre la qualité de l'offre alimentaire.

Le projet s'appuie principalement sur :

- Instacart Market Basket Analysis ;
- Open Food Facts ;
- RappelConso.

---

## Problématique

Comment transformer des données hétérogènes en indicateurs et prédictions fiables pour orienter les décisions commerciales de Panier Vert et améliorer le suivi de son offre alimentaire ?

---

## Objectifs métier

### Segmentation client

Décrire les clients selon :

- leur fréquence d'achat ;
- la composition de leurs paniers ;
- leur comportement de consommation.

### Analyse d'associations

Identifier les produits fréquemment achetés ensemble afin de :

- proposer des offres groupées ;
- améliorer les recommandations ;
- optimiser l'organisation des rayons.

### Prédiction de réachat

Estimer la probabilité qu'un produit soit commandé à nouveau lors d'une prochaine commande.

### Veille qualité

Mettre à disposition :

- les informations nutritionnelles ;
- les alertes sanitaires ;
- les rappels produits.

---

## Architecture

L'architecture détaillée est disponible dans :

```text
docs/architecture.png
```

Schéma d'architecture : voir le fichier `docs/architecture.png`. L'architecture couvre les sources de données, l'ingestion, le stockage, le prétraitement, le data mining et la restitution métier. imagearchitecture.png†image_0 (Image: architecture fonctionnelle Panier Vert)

---

## Technologies retenues

- Python
- Pandas
- Scikit-Learn
- MLflow
- DVC
- GitHub Actions
- BentoML
- Docker
- Evidently
- Streamlit

---

## Structure du projet

```text
paniervert-mspr/
│
├── docs/
│   ├── architecture.png
│   ├── architecture.md
│   └── cadrage.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   │   ├── instacart.py
│   │   ├── openfoodfacts.py
│   │   ├── rappelconso.py
│   │   └── scraping.py
│   │
│   ├── processing/
│   │
│   ├── mining/
│   │
│   └── pipeline.py
│
├── dashboard/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Description des composants

#### ingestion

- `instacart.py` : chargement des données historiques de commandes Instacart.
- `openfoodfacts.py` : collecte des données nutritionnelles Open Food Facts.
- `rappelconso.py` : collecte des alertes sanitaires RappelConso.
- `scraping.py` : collecte complémentaire de données produits.

#### processing

Préparation, nettoyage, enrichissement et contrôles qualité des données.

#### mining

Segmentation client, règles d'association et modèle de réachat.

#### pipeline.py

Orchestration des différentes étapes du traitement.

---

## Méthodologie

### Collecte et préparation

- Collecte automatisée des données.
- Stockage des données brutes au format Parquet.
- Séparation entre données brutes et données préparées.
- Journalisation des traitements.
- Gestion des erreurs temporaires.
- Contrôles qualité sur :
  - le schéma ;
  - la complétude ;
  - les doublons ;
  - les anomalies ;
  - la fraîcheur des données.

### Rapprochement des produits

Correspondance des catalogues à partir :

- des codes-barres ;
- des libellés ;
- des marques ;
- du conditionnement.

### Data Mining

- Segmentation client par K-Means.
- Étude comparative avec DBSCAN.
- Analyse des associations produits avec FP-Growth.
- Évaluation des règles via le support, la confiance et le lift.

### Modèle de réachat

- Comparaison avec une approche basée sur la fréquence historique.
- Construction des jeux d'apprentissage à partir du seul historique disponible avant la commande cible.
- Interprétation des prédictions grâce à SHAP.

### Industrialisation

- Tracking des expérimentations avec MLflow.
- Versionnement des données et pipelines avec DVC.
- Automatisation des contrôles avec GitHub Actions.
- Déploiement via BentoML.
- Conteneurisation Docker.
- Surveillance de la dérive avec Evidently.
- Restitution via Streamlit.

---

## Roadmap

### J1

- [x] Cadrage du projet
- [x] Architecture cible
- [x] Dépôt GitHub
- [x] Structure du projet
- [ ] Collecte des données
- [ ] Contrôles qualité
- [ ] Segmentation client
- [ ] Analyse d'associations
- [ ] Modèle de réachat
- [ ] Dashboard métier

---

## Docker

### Prérequis

- Docker Desktop installé
- Docker Compose disponible

### Construire les images

```bash
docker compose build
```

### Lancer tout le projet

```bash
docker compose up
```

Cette commande lance :

- le pipeline de nettoyage dans le service `paniervert` ;
- le dashboard Streamlit dans le service `streamlit`.

Le dashboard est accessible sur :

```text
http://localhost:8501
```

### Lancer uniquement les commandes utiles

Pipeline :

```bash
docker compose run --rm paniervert
```

Dashboard seul :

```bash
docker compose up streamlit
```

Arrêter les conteneurs :

```bash
docker compose down
```

### Données locales

Le dossier local `./data` est monté dans `/app/data` dans les conteneurs. Les données brutes et préparées restent disponibles hors du conteneur et ne doivent pas être versionnées dans Git.

---

## Équipe

Projet réalisé dans le cadre de la MSPR Data Mining.
