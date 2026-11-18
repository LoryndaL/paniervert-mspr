from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


st.set_page_config(page_title="Panier Vert - Dashboard", layout="wide")
st.title("Panier Vert - Données nettoyées")

processed_files = sorted(PROCESSED_DIR.rglob("*.parquet"))
if not processed_files:
    st.warning("Aucune donnée nettoyée trouvée. Lancez d'abord `python src/pipeline.py`.")
    st.stop()

labels = [str(path.relative_to(PROCESSED_DIR)) for path in processed_files]
selection = st.sidebar.selectbox("Table nettoyée", labels)
selected_path = PROCESSED_DIR / selection

df = pd.read_parquet(selected_path)
st.caption(f"Fichier : `{selection}`")
st.metric("Lignes", f"{len(df):,}")
st.metric("Colonnes", len(df.columns))
st.dataframe(df.head(100), use_container_width=True)

with st.expander("Liste des colonnes"):
    st.write(list(df.columns))
