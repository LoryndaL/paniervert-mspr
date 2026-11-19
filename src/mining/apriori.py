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
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROCESSED_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_order_products_path() -> Path | None:
    """Localise la table des lignes de commande prior (Silver / Processed / Raw)."""
    candidates = [
        SILVER_DIR / "order_products__prior.parquet",
        PROCESSED_DIR / "order_products__prior.parquet",
        PROCESSED_DIR / "instacart" / "order_products__prior.parquet",
        RAW_DIR / "instacart" / "order_products__prior.parquet",
        RAW_DIR / "order_products__prior.parquet",
        RAW_DIR / "instacart" / "order_products__prior.csv",
        RAW_DIR / "order_products__prior.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def resolve_products_path() -> Path | None:
    """Localise le référentiel produits (Silver / Processed / Raw)."""
    candidates = [
        SILVER_DIR / "products.parquet",
        PROCESSED_DIR / "products_enriched.parquet",
        PROCESSED_DIR / "products.parquet",
        PROCESSED_DIR / "instacart" / "products.parquet",
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


def build_product_labels(products_df: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, bool]]:
    """Construit un libellé produit lisible (sans ID brut) et un indicateur bio."""
    label_map: dict[str, str] = {}
    bio_map: dict[str, bool] = {}

    if products_df is None or "product_id" not in products_df.columns:
        return label_map, bio_map

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
        raw_name = str(row[name_col]).strip() if (name_col and pd.notna(row[name_col])) else ""
        base_name = raw_name if raw_name else f"Article {pid}"
        is_bio = (
            bool(row[bio_col])
            if (bio_col and pd.notna(row[bio_col]))
            else ("organic" in base_name.lower() or "bio" in base_name.lower())
        )
        bio_map[pid] = is_bio
        label_map[pid] = f"[BIO] {base_name}" if is_bio else base_name

    return label_map, bio_map


def prepare_baskets(
    op_df: pd.DataFrame,
    products_df: pd.DataFrame | None,
    top_n_items: int = 1500,
    max_orders: int | None = 120_000,
) -> tuple[list[list[str]], dict[str, bool]]:
    """Filtre les top N produits les plus vendus et forme les paniers par commande."""
    if "order_id" not in op_df.columns or "product_id" not in op_df.columns:
        raise ValueError("Les colonnes 'order_id' et 'product_id' sont requises.")

    work = op_df[["order_id", "product_id"]].dropna().copy()
    work["product_id"] = work["product_id"].astype(str)

    if max_orders is not None:
        unique_orders = work["order_id"].drop_duplicates()
        if len(unique_orders) > max_orders:
            sampled = unique_orders.sample(n=max_orders, random_state=42)
            work = work[work["order_id"].isin(sampled)]
            logging.info("Échantillon de %d commandes retenu.", max_orders)

    top_pids = work["product_id"].value_counts().head(top_n_items).index
    work = work[work["product_id"].isin(top_pids)]

    label_map, bio_map = build_product_labels(products_df)
    work["item_label"] = work["product_id"].map(lambda pid: label_map.get(pid, f"Article {pid}"))
    item_is_bio = {label_map.get(pid, f"Article {pid}"): is_bio for pid, is_bio in bio_map.items()}

    baskets = (
        work.groupby("order_id")["item_label"]
        .apply(lambda s: sorted(set(s.tolist())))
        .tolist()
    )
    baskets = [b for b in baskets if len(b) >= 2]
    return baskets, item_is_bio


def extract_rules_with_mlxtend(
    baskets: list[list[str]],
    min_support: float = 0.005,
    min_confidence: float = 0.20,
    min_lift: float = 1.2,
) -> pd.DataFrame | None:
    """Utilise mlxtend (TransactionEncoder sparse + FP-Growth + association_rules)."""
    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        return None

    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets, sparse=True)
    ohe = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    freq = fpgrowth(ohe, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq, num_itemsets=len(ohe), metric="lift", min_threshold=min_lift)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["confidence"] >= min_confidence].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: sorted(list(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: sorted(list(s)))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]


def extract_rules_fallback(
    baskets: list[list[str]],
    min_support: float = 0.005,
    min_confidence: float = 0.20,
    min_lift: float = 1.2,
    max_len: int = 3,
) -> pd.DataFrame:
    """Repli léger sans mlxtend (itemsets fréquents + règles support/confiance/lift)."""
    n_baskets = len(baskets)
    if n_baskets == 0:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    basket_sets = [frozenset(b) for b in baskets]
    min_count = max(2, int(np.ceil(min_support * n_baskets)))

    c1: Counter[str] = Counter()
    for b in basket_sets:
        c1.update(b)

    freq_sets: dict[frozenset[str], int] = {
        frozenset([item]): cnt for item, cnt in c1.items() if cnt >= min_count
    }
    current = set(freq_sets.keys())

    for length in range(2, max_len + 1):
        pool = sorted({it for s in current for it in s})
        next_freq: dict[frozenset[str], int] = {}
        for comb in combinations(pool, length):
            cand = frozenset(comb)
            if all(frozenset(sub) in current for sub in combinations(comb, length - 1)):
                cnt = sum(1 for b in basket_sets if cand.issubset(b))
                if cnt >= min_count:
                    next_freq[cand] = cnt
        if not next_freq:
            break
        freq_sets.update(next_freq)
        current = set(next_freq.keys())

    records: list[dict] = []
    for iset, cnt_both in freq_sets.items():
        if len(iset) < 2:
            continue
        sup_both = cnt_both / n_baskets
        items_sorted = sorted(list(iset))
        for r in range(1, len(items_sorted)):
            for ant_t in combinations(items_sorted, r):
                ant = frozenset(ant_t)
                cons = iset - ant
                if not cons or ant not in freq_sets or cons not in freq_sets:
                    continue
                conf = sup_both / (freq_sets[ant] / n_baskets)
                lift = conf / (freq_sets[cons] / n_baskets)
                if conf >= min_confidence and lift >= min_lift:
                    records.append(
                        {
                            "antecedents": sorted(list(ant)),
                            "consequents": sorted(list(cons)),
                            "support": float(sup_both),
                            "confidence": float(conf),
                            "lift": float(lift),
                        }
                    )

    if not records:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return (
        pd.DataFrame(records)
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .reset_index(drop=True)
    )


def build_business_recommendations(rules_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Traduit les top règles en recommandations chiffrées (vente croisée, rayon, lot)."""
    recs: list[dict] = []
    actions = [
        "Offre groupée / bundle promotionnel",
        "Vente croisée au passage panier",
        "Implantation / mise en avant en rayon",
    ]
    for idx, row in rules_df.head(top_n).iterrows():
        ant_txt = " + ".join(row["antecedents"])
        cons_txt = " + ".join(row["consequents"])
        sup_pct = round(float(row["support"]) * 100, 2)
        conf_pct = round(float(row["confidence"]) * 100, 1)
        lift_val = round(float(row["lift"]), 2)
        action_type = actions[idx % len(actions)]

        recs.append(
            {
                "rang": idx + 1,
                "regle": f"{ant_txt} -> {cons_txt}",
                "type_action": action_type,
                "support_pct": sup_pct,
                "confiance_pct": conf_pct,
                "lift": lift_val,
                "recommandation": (
                    f"{action_type} : {sup_pct}% des paniers associent '{ant_txt}' et '{cons_txt}' "
                    f"({conf_pct}% des acheteurs de '{ant_txt}' prennent '{cons_txt}', lift {lift_val}x)."
                ),
            }
        )
    return recs


def main(
    min_support: float = 0.005,
    min_confidence: float = 0.20,
    min_lift: float = 1.2,
    top_n_items: int = 1500,
) -> Path | None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    op_path = resolve_order_products_path()
    if op_path is None:
        logging.error("Fichier order_products__prior introuvable.")
        return None

    products_path = resolve_products_path()
    products_df = load_table(products_path) if products_path is not None else None

    logging.info("Chargement de %s...", op_path)
    op_df = load_table(op_path)
    baskets, item_is_bio = prepare_baskets(op_df, products_df, top_n_items=top_n_items)

    if len(baskets) < 10:
        logging.warning("Pas assez de paniers (%d) pour extraire des règles.", len(baskets))
        return None

    rules_df = extract_rules_with_mlxtend(
        baskets=baskets,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    engine_used = "mlxtend_fpgrowth"
    if rules_df is None:
        rules_df = extract_rules_fallback(
            baskets=baskets,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )
        engine_used = "native_fallback"

    if not rules_df.empty:
        rules_df["support"] = rules_df["support"].round(4)
        rules_df["confidence"] = rules_df["confidence"].round(4)
        rules_df["lift"] = rules_df["lift"].round(4)
        rules_df["antecedents_label"] = rules_df["antecedents"].apply(lambda xs: " + ".join(xs))
        rules_df["consequents_label"] = rules_df["consequents"].apply(lambda xs: " + ".join(xs))
        rules_df["is_bio_related"] = rules_df.apply(
            lambda r: any(item_is_bio.get(x, False) for x in (r["antecedents"] + r["consequents"])),
            axis=1,
        )

    gold_output = GOLD_DIR / "association_rules.parquet"
    rules_df.to_parquet(gold_output, index=False)
    logging.info("Règles enregistrées en zone Gold : %s (%d règles)", gold_output, len(rules_df))

    top_10_recs = build_business_recommendations(rules_df, top_n=10)
    print(f"\n=== TOP 10 RÈGLES D'ASSOCIATION ({engine_used}) ===")
    for item in top_10_recs:
        print(f"#{item['rang']} | {item['recommandation']}")

    summary_payload = {
        "engine": engine_used,
        "baskets_analyzed": int(len(baskets)),
        "top_n_products_kept": int(top_n_items),
        "parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
        "rules_count": int(len(rules_df)),
        "bio_related_rules_count": int(rules_df["is_bio_related"].sum()) if "is_bio_related" in rules_df.columns else 0,
        "top_10_recommendations": top_10_recs,
    }
    summary_path = MODELS_DIR / "association_rules_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Synthèse enregistrée : %s", summary_path)

    return gold_output


if __name__ == "__main__":
    main()