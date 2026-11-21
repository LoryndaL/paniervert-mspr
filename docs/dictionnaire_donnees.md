# Dictionnaire des données brutes

## Instacart

| Fichier | Colonnes principales | Description |
|---|---|---|
| `orders.parquet` | `order_id`, `user_id`, `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order` | commandes passées par les clients |
| `products.parquet` | `product_id`, `product_name`, `aisle_id`, `department_id` | catalogue des produits |
| `order_products__prior.parquet` | `order_id`, `product_id`, `add_to_cart_order`, `reordered` | produits présents dans les anciennes commandes |
| `order_products__train.parquet` | `order_id`, `product_id`, `add_to_cart_order`, `reordered` | produits présents dans les commandes d'entraînement |
| `aisles.parquet` | `aisle_id`, `aisle` | libellés des allées |
| `departments.parquet` | `department_id`, `department` | libellés des départements |

## Open Food Facts

`data/raw/openfoodfacts.parquet` contient notamment :

| Colonne | Description |
|---|---|
| `code` | code-barres du produit |
| `product_name` | nom du produit |
| `brands` | marque |
| `categories` | catégories commerciales |
| `countries_tags` | pays associés |
| `nutriscore_grade` | Nutri-Score |
| `nova_group` | niveau NOVA |
| `ecoscore_grade` | Eco-Score |
| `ingredients_text` | ingrédients |
| `allergens_tags` | allergènes |
| `additives_tags` | additifs |
| `energy-kcal_100g`, `fat_100g`, `sugars_100g`, `proteins_100g`, `salt_100g` | valeurs nutritionnelles pour 100 g |

## RappelConso

`data/raw/rappelconso.parquet` contient les champs exportés par la source officielle, notamment les identifiants GTIN, les motifs, les risques, les dates et les modalités de rappel. Le schéma exact est vérifié à l'ouverture du fichier avec `pyarrow` ou Pandas.

## Scraping agriculture européenne

- `organic_logo_page.html` : copie HTML brute de la page collectée.
- `organic_logo_data.json` : titre, sous-titres, paragraphes, liens et sections structurées.
- `source_url` : URL d'origine.
- `fetched_at` : horodatage UTC de la collecte.

## Métadonnées de collecte

`data/raw/collection_metadata.json` contient :

- date globale de collecte en UTC;
- nom de chaque source;
- date de début et de fin;
- état `ok` ou `error`;
- message d'erreur éventuel.
