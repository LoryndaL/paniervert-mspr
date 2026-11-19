from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_order_products_path() -> Path | None:
    """Localise la table des produits par commande (parquet ou csv)."""
    candidates = [
        PROCESSED_DIR / "instacart" / "order_products__prior.parquet",
        PROCESSED_DIR / "order_products__prior.parquet",
        RAW_DIR / "instacart" / "order_products__prior.parquet",
        RAW_DIR / "order_products__prior.parquet",
        RAW_DIR / "instacart" / "order_products__prior.csv",
        RAW_DIR / "order_products__prior.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def resolve_products_path() -> Path | None:
    """Localise le catalogue produits enrichi ou brut."""
    candidates = [
        PROCESSED_DIR / "products_enriched.parquet",
        PROCESSED_DIR / "instacart" / "products.parquet",
        PROCESSED_DIR / "products.parquet",
        RAW_DIR / "instacart" / "products.parquet",
        RAW_DIR / "products.parquet",
        RAW_DIR / "instacart" / "products.csv",
        RAW_DIR / "products.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_table(path: Path) -> pd.DataFrame:
    """Charge un fichier Parquet ou CSV."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_baskets(
    op_df: pd.DataFrame,
    products_df: pd.DataFrame | None,
    max_orders: int = 80_000,
    top_n_items: int = 200,
) -> tuple[list[frozenset[str]], dict[str, bool]]:
    """Construit la liste des paniers (order_id -> ensemble d'items) et repère les articles bio."""
    if "order_id" not in op_df.columns or "product_id" not in op_df.columns:
        raise ValueError("Les colonnes 'order_id' et 'product_id' sont requises pour Apriori.")

    work = op_df[["order_id", "product_id"]].dropna().copy()
    work["product_id"] = work["product_id"].astype(str)

    unique_orders = work["order_id"].drop_duplicates()
    if len(unique_orders) > max_orders:
        sampled_orders = unique_orders.sample(n=max_orders, random_state=42)
        work = work[work["order_id"].isin(sampled_orders)]
        logging.info("Échantillon de %d commandes retenu pour Apriori.", max_orders)

    top_pids = work["product_id"].value_counts().head(top_n_items).index
    work = work[work["product_id"].isin(top_pids)]

    label_map: dict[str, str] = {}
    bio_map: dict[str, bool] = {}

    if products_df is not None and "product_id" in products_df.columns:
        pwork = products_df.copy()
        pwork["product_id"] = pwork["product_id"].astype(str)
        name_col = next(
            (c for c in ["product_name_clean", "product_name", "name"] if c in pwork.columns),
            None,
        )
        bio_col = next(
            (c for c in ["is_eu_organic_certified", "is_organic", "organic"] if c in pwork.columns),
            None,
        )
        for _, row in pwork.iterrows():
            pid = row["product_id"]
            if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip():
                base_name = str(row[name_col]).strip()
            else:
                base_name = f"Produit_{pid}"

            is_bio = (
                bool(row[bio_col])
                if (bio_col and pd.notna(row[bio_col]))
                else ("organic" in base_name.lower() or "bio" in base_name.lower())
            )
            bio_map[pid] = is_bio
            label_map[pid] = f"[BIO] {base_name}" if is_bio else base_name

    work["item_label"] = work["product_id"].map(lambda pid: label_map.get(pid, f"Produit_{pid}"))
    item_is_bio = {
        label_map.get(pid, f"Produit_{pid}"): is_bio for pid, is_bio in bio_map.items()
    }

    grouped = work.groupby("order_id")["item_label"].apply(lambda s: frozenset(s.unique()))
    baskets = [b for b in grouped.tolist() if len(b) >= 2]
    return baskets, item_is_bio


def mine_association_rules(
    baskets: list[frozenset[str]],
    min_support: float = 0.01,
    min_confidence: float = 0.25,
    min_lift: float = 1.1,
    max_len: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Extrait les itemsets fréquents et les règles (support, confiance, lift)."""
    n_baskets = len(baskets)
    if n_baskets == 0:
        return [], []

    min_count = max(2, int(np.ceil(min_support * n_baskets)))

    c1: Counter[str] = Counter()
    for b in baskets:
        c1.update(b)

    freq_sets: dict[frozenset[str], int] = {
        frozenset([item]): cnt for item, cnt in c1.items() if cnt >= min_count
    }
    current_candidates = set(freq_sets.keys())

    for length in range(2, max_len + 1):
        items_pool = sorted({item for s in current_candidates for item in s})
        next_freq: dict[frozenset[str], int] = {}

        for comb in combinations(items_pool, length):
            cand = frozenset(comb)
            if all(frozenset(sub) in current_candidates for sub in combinations(comb, length - 1)):
                cnt = sum(1 for b in baskets if cand.issubset(b))
                if cnt >= min_count:
                    next_freq[cand] = cnt

        if not next_freq:
            break
        freq_sets.update(next_freq)
        current_candidates = set(next_freq.keys())

    itemsets_records = [
        {
            "items": sorted(list(iset)),
            "length": len(iset),
            "support": round(cnt / n_baskets, 4),
            "count": int(cnt),
        }
        for iset, cnt in sorted(freq_sets.items(), key=lambda kv: kv[1], reverse=True)
    ]

    rules_records: list[dict] = []
    for iset, cnt_both in freq_sets.items():
        if len(iset) < 2:
            continue
        sup_both = cnt_both / n_baskets
        items_list = sorted(list(iset))

        for r in range(1, len(items_list)):
            for ant_tuple in combinations(items_list, r):
                antecedent = frozenset(ant_tuple)
                consequent = iset - antecedent
                if not consequent or antecedent not in freq_sets or consequent not in freq_sets:
                    continue

                sup_ant = freq_sets[antecedent] / n_baskets
                sup_cons = freq_sets[consequent] / n_baskets
                if sup_ant <= 0 or sup_cons <= 0:
                    continue

                confidence = sup_both / sup_ant
                lift = confidence / sup_cons

                if confidence >= min_confidence and lift >= min_lift:
                    rules_records.append(
                        {
                            "antecedents": sorted(list(antecedent)),
                            "consequents": sorted(list(consequent)),
                            "support": round(float(sup_both), 4),
                            "confidence": round(float(confidence), 4),
                            "lift": round(float(lift), 4),
                        }
                    )

    rules_records.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
    return itemsets_records, rules_records


def main(
    min_support: float = 0.01,
    min_confidence: float = 0.25,
    min_lift: float = 1.1,
) -> Path | None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    op_path = resolve_order_products_path()
    if op_path is None:
        logging.error("Fichier order_products__prior introuvable pour l'analyse Apriori.")
        return None

    products_path = resolve_products_path()
    products_df = load_table(products_path) if products_path is not None else None

    logging.info("Chargement de %s...", op_path)
    op_df = load_table(op_path)
    baskets, item_is_bio = build_baskets(op_df, products_df)

    if len(baskets) < 10:
        logging.warning("Pas assez de paniers (%d) pour extraire des règles d'association.", len(baskets))
        return None

    logging.info("Extraction des règles Apriori sur %d paniers...", len(baskets))
    itemsets, rules = mine_association_rules(
        baskets=baskets,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )

    print(f"\n=== RÈGLES APRIORI DÉTECTÉES ({len(rules)} règles) ===")
    for r in rules[:10]:
        ant_str = " + ".join(r["antecedents"])
        cons_str = " + ".join(r["consequents"])
        print(
            f"{ant_str}  -->  {cons_str} | "
            f"support={r['support']:.4f} | confiance={r['confidence']:.4f} | lift={r['lift']:.2f}"
        )

    bio_rules = [
        r
        for r in rules
        if any(item_is_bio.get(x, False) for x in r["antecedents"] + r["consequents"])
    ]

    rules_parquet_path = GOLD_DIR / "apriori_rules.parquet"
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df["antecedents_str"] = rules_df["antecedents"].apply(lambda xs: " + ".join(xs))
        rules_df["consequents_str"] = rules_df["consequents"].apply(lambda xs: " + ".join(xs))
    rules_df.to_parquet(rules_parquet_path, index=False)
    logging.info("Règles Apriori sauvegardées : %s", rules_parquet_path)

    report_payload = {
        "baskets_analyzed": int(len(baskets)),
        "parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
        "frequent_itemsets_count": int(len(itemsets)),
        "rules_count": int(len(rules)),
        "bio_related_rules_count": int(len(bio_rules)),
        "top_itemsets": itemsets[:30],
        "top_rules": rules[:50],
        "top_bio_rules": bio_rules[:25],
    }
    report_path = MODELS_DIR / "apriori_rules.json"
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Rapport Apriori sauvegardé : %s", report_path)

    return rules_parquet_path


if __name__ == "__main__":
    main()