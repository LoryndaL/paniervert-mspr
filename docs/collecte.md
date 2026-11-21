# Collecte des données

## Commande reproductible

Depuis la racine du projet :

```bash
make collect
```

La collecte lance les sources Instacart, Open Food Facts, RappelConso et agriculture.ec.europa.eu. Les journaux et la commande utilisée sont visibles dans le terminal.

## Sources et conformité

| Source | Usage | Licence / cadre | Traitement |
|---|---|---|---|
| Instacart Market Basket Analysis | commandes et catalogue | jeu Kaggle public ; identifiants Kaggle nécessaires | télécharge les CSV puis les convertit en Parquet |
| Open Food Facts | informations produits | base ouverte ; données mises à disposition publiquement | téléchargement distant, filtre France, export Parquet |
| RappelConso | alertes sanitaires | plateforme data.gouv / économie.gouv | export Parquet |
| Commission européenne — logo bio | contexte documentaire | robots.txt et CGU vérifiés avant collecte | sauvegarde HTML et JSON |

## RGPD et minimisation

- Les données sont collectées pour les finalités de segmentation, recommandation et veille sanitaire.
- Les identifiants techniques sont conservés séparément des données analytiques lorsque possible.
- Aucun objectif publicitaire individuel ou décision automatisée sensible n'est appliqué.
- Les données brutes ne sont pas modifiées après collecte.

## Fiabilité

- backoff exponentiel sur les erreurs réseau;
- reprises sur HTTP 429, 5xx et erreurs de transport;
- respect de `Retry-After` lorsque fourni;
- journalisation de l'état de chaque source dans `data/raw/collection_metadata.json`;
- code de sortie non nul si une source échoue.

## Scraping

Le script vérifie `https://agriculture.ec.europa.eu/robots.txt`, applique un identifiant HTTP explicite, respecte un délai limité aux requêtes effectuées et ne contourne aucun mécanisme d'accès.
