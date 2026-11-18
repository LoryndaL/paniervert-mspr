from ingestion.instacart import main as instacart
from ingestion.openfoodfacts import main as openfoodfacts
from ingestion.rappelconso import main as rappelconso
from ingestion.scraping import main as scraping

from preprocessing.clean_data import main as clean_data
from preprocessing.merge_data import main as merge_data

from features.build_features import main as build_features

from mining.clustering import main as clustering
from mining.apriori import main as apriori
from mining.classification import main as classification


def main():
    print("1. Collecte")
    instacart()
    openfoodfacts()
    rappelconso()
    scraping()

    print("2. Nettoyage")
    clean_data()

    print("3. Fusion")
    merge_data()

    print("4. Feature Engineering")
    build_features()

    print("5. Clustering")
    clustering()

    print("6. Apriori")
    apriori()

    print("7. Classification")
    classification()

    print("Pipeline terminée ✅")


if __name__ == "__main__":
    main()
