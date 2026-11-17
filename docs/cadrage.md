# Cadrage du projet – Panier Vert

## Contexte

Panier Vert est un réseau fictif de 42 épiceries bio disposant d'un service de commande en ligne et de retrait en magasin.

La direction commerciale souhaite mieux exploiter ses historiques de commandes ainsi que des données publiques concernant les produits afin de :

- fidéliser les clients ;
- améliorer la pertinence des recommandations ;
- suivre la qualité de l'offre alimentaire ;
- disposer d'indicateurs fiables pour la prise de décision.

Le projet s'appuie sur plusieurs sources de données :

- Instacart Market Basket Analysis (historique des commandes) ;
- Open Food Facts (informations nutritionnelles des produits) ;
- RappelConso (rappels et alertes sanitaires officielles) ;
- sources web complémentaires obtenues par scraping.

---

# Problématique

Comment transformer des données hétérogènes en indicateurs et prédictions fiables pour orienter les décisions commerciales de Panier Vert et améliorer le suivi de son offre alimentaire ?

---

# Parties prenantes

## Direction commerciale

Utilise les analyses pour :

- cibler les campagnes marketing ;
- améliorer les recommandations produits ;
- suivre les tendances de consommation ;
- identifier les opportunités commerciales.

## Équipe Data

Utilise la plateforme pour :

- rejouer les traitements ;
- contrôler la qualité des données ;
- entraîner et évaluer les modèles ;
- réutiliser les résultats analytiques.

## Utilisateurs métier

Consultent les résultats via un tableau de bord sécurisé.

---

# Objectifs métier

## 1. Segmentation client

Identifier des groupes de clients présentant des comportements similaires selon :

- leur fréquence d'achat ;
- leur volume de commandes ;
- la composition de leurs paniers.

### Résultat attendu

Production de segments exploitables pour des actions marketing ciblées.

---

## 2. Analyse des associations produits

Identifier les produits fréquemment achetés ensemble.

### Résultat attendu

Mise en évidence de règles d'association permettant :

- la création d'offres groupées ;
- l'amélioration des recommandations ;
- l'optimisation du merchandising.

---

## 3. Prédiction de réachat

Estimer la probabilité qu'un client rachète un produit déjà acheté lors d'une prochaine commande.

### Résultat attendu

Priorisation des recommandations personnalisées.

---

## 4. Veille qualité et sécurité alimentaire

Croiser le catalogue produit avec :

- les données nutritionnelles ;
- les alertes et rappels sanitaires.

### Résultat attendu

Faciliter le suivi de la qualité de l'offre et des éventuels risques sanitaires.

---

# Architecture fonctionnelle

Le système s'organise autour des étapes suivantes :

1. Collecte des données
2. Stockage des données brutes
3. Prétraitement et enrichissement
4. Analyse Data Mining
5. Exposition des résultats
6. Consultation via tableau de bord

Le schéma détaillé est disponible dans :

```text
docs/architecture.png
```

Architecture générale de la solution : Sources → Ingestion → Stockage → Prétraitement → Data Mining → Dashboard → Utilisateurs. imagearchitecture.png†image_0 (Image: architecture fonctionnelle Panier Vert)

---

# Méthodologie

## Collecte et préparation

### Sources

- Instacart
- Open Food Facts
- RappelConso
- Sources web complémentaires

### Exigences

- automatisation des collectes ;
- stockage des données brutes au format Parquet ;
- séparation des données brutes et nettoyées ;
- journalisation des traitements ;
- reprise des erreurs temporaires ;
- contrôle qualité systématique.

### Contrôles qualité

- conformité du schéma ;
- complétude ;
- doublons ;
- anomalies ;
- fraîcheur des données.

---

## Rapprochement des produits

Ordre de priorité :

1. correspondance par code-barres ;
2. correspondance par libellé ;
3. correspondance par marque ;
4. correspondance par conditionnement.

### Indicateurs à produire

- taux de couverture ;
- précision sur un échantillon validé ;
- taux de correspondance.

---

## Analyse Data Mining

### Segmentation

Algorithmes étudiés :

- K-Means ;
- DBSCAN.

### Analyse d'associations

Algorithmes :

- FP-Growth ;
- éventuellement Apriori.

Indicateurs :

- support ;
- confiance ;
- lift.

---

## Modèle de réachat

Objectif :

Prédire la probabilité de réachat d'un produit à partir de l'historique client.

Approches comparées :

- classement par fréquence historique ;
- modèle de classification supervisée.

Explicabilité :

- SHAP.

---

## Industrialisation

### Gestion des expériences

- MLflow

### Gestion des données

- DVC

### Automatisation

- GitHub Actions

### Déploiement

- Docker
- BentoML

### Monitoring

- Evidently

### Restitution

- Streamlit

---

# Livrables attendus

## Données

- données brutes ;
- données préparées ;
- rapports qualité.

## Analyses

- segmentation clients ;
- règles d'association ;
- modèle de réachat.

## Industrialisation

- pipeline automatisé ;
- suivi MLflow ;
- pipeline DVC ;
- CI/CD GitHub Actions ;
- conteneur Docker ;
- service BentoML.

## Restitution

- tableau de bord Streamlit.
