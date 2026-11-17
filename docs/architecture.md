# Architecture de la solution

## Vue d'ensemble

L'architecture de la solution est représentée dans le fichier :

- `architecture.png`

Le schéma présente le flux complet de traitement, depuis la collecte des données jusqu'à leur restitution aux utilisateurs métiers.

## Schéma d'architecture

Voir le fichier :

```text
docs/architecture.png
```

## Sources de données

Les données exploitées proviennent de plusieurs sources :

- Instacart Market Basket Analysis (historique des commandes)
- Open Food Facts (informations nutritionnelles)
- RappelConso (rappels sanitaires officiels)
- Sources web complémentaires via scraping

## Flux de traitement

### 1. Collecte des données

Collecte automatisée depuis :

- Instacart
- Open Food Facts
- RappelConso
- Sources web complémentaires

### 2. Ingestion

Les données sont récupérées et intégrées à l'aide de scripts Python dédiés.

Fonctionnalités prévues :

- gestion des erreurs ;
- journalisation des traitements ;
- reprise sur incident ;
- limitation du débit des appels API.

### 3. Stockage

Les données brutes sont conservées dans un espace dédié au format Parquet.

Objectifs :

- séparation des données brutes et préparées ;
- optimisation des performances de lecture ;
- traçabilité des traitements.

### 4. Prétraitement et enrichissement

Les données sont :

- nettoyées ;
- dédoublonnées ;
- contrôlées ;
- enrichies avec les informations produits disponibles.

Des contrôles qualité sont réalisés sur :

- le schéma ;
- la complétude ;
- les doublons ;
- la fraîcheur ;
- les anomalies.

### 5. Data Mining

#### Segmentation client

Algorithmes étudiés :

- K-Means ;
- DBSCAN.

Objectif :

- identifier des groupes de clients ayant des comportements d'achat similaires.

#### Analyse d'associations

Algorithmes étudiés :

- FP-Growth ;
- Apriori (si nécessaire).

Indicateurs calculés :

- support ;
- confiance ;
- lift.

Objectif :

- identifier les produits fréquemment achetés ensemble.

#### Prédiction de réachat

Objectif :

- estimer la probabilité qu'un client rachète un produit lors d'une prochaine commande.

### 6. Restitution

Les résultats sont présentés dans un tableau de bord regroupant :

- les KPI commerciaux ;
- les segments clients ;
- les recommandations produits ;
- les alertes sanitaires ;
- les indicateurs qualité.

### 7. Utilisateurs

Les résultats sont exploités par :

- la direction commerciale ;
- les équipes métier ;
- l'équipe Data.

---

## Industrialisation prévue

### Gestion des expériences

- MLflow

### Versionnement des données

- DVC

### Intégration continue

- GitHub Actions

### Déploiement

- Docker
- BentoML

### Monitoring

- Evidently

### Restitution

- Streamlit

---

## Technologies retenues

| Domaine | Technologies |
|----------|-------------|
| Langage | Python |
| Analyse de données | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Data Mining | K-Means, DBSCAN, FP-Growth |
| Tracking | MLflow |
| Versionnement des données | DVC |
| CI/CD | GitHub Actions |
| Déploiement | Docker, BentoML |
| Monitoring | Evidently |
| Dashboard | Streamlit |

---

## Évolutions prévues

- Automatisation complète des traitements.
- Collecte incrémentale quotidienne des alertes RappelConso.
- Déploiement conteneurisé.
- Suivi des performances des modèles dans le temps.
- Mise à disposition d'un tableau de bord décisionnel.
