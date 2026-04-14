import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="La Liga Dashboard", layout="wide")

# ================================
# LOAD FUNCTIONS
# ================================
def load_df(file):
    df = pd.read_csv(file, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

def prepare_df(df):
    df = df.copy()
    lc = {c.lower(): c for c in df.columns}

    # Score parsing
    if "predict score" in lc and "actual score" in lc:
        ps, ac = lc["predict score"], lc["actual score"]

        def parse(s):
            try:
                h, a = str(s).split("-")
                return int(h), int(a)
            except:
                return None, None

        df[["ph","pa"]] = pd.DataFrame(df[ps].apply(parse).tolist())
        df[["ah","aa"]] = pd.DataFrame(df[ac].apply(parse).tolist())

        df["pred_total"] = df["ph"] + df["pa"]
        df["act_total"]  = df["ah"] + df["aa"]
        df["residual"]   = df["act_total"] - df["pred_total"]

    # Outcome
    if "predict outcome" in lc and "actual outcome" in lc:
        po, ao = lc["predict outcome"], lc["actual outcome"]
        df["correct"] = df[po] == df[ao]

    # Top-N
    for k, new in [("1 from top 5","top5"),("1 from top 3","top3"),("1 from top 2","top2")]:
        if k in lc:
            df[new] = df[lc[k]].str.upper() == "YES"

    return df

# ================================
# SIDEBAR (IMPORTANT FIRST)
# ================================
with st.sidebar:
    st.title("⚽ Dashboard")
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type="csv",
        accept_multiple_files=True
    )

# ================================
# MODEL LOADING (FIXED)
# ================================
models = {}

if uploaded_files:
    for i, f in enumerate(uploaded_files):
        raw = load_df(f)
        df  = prepare_df(raw)

        # ✅ FIX: unique model names
        name = f.name.replace(".csv", "")
        unique_name = f"{name} ({i+1})"

        models[unique_name] = df

# ================================
# NO FILE HANDLING
# ================================
if not models:
    st.warning("Upload at least one CSV file")
    st.stop()

# ================================
# MODEL SELECTOR
# ================================
with st.sidebar:
    selected_model = st.selectbox("Select Model", list(models.keys()))

df = models[selected_model]

# ================================
# HEADER
# ================================
st.title("⚽ La Liga Prediction Dashboard")
st.write(f"Model: {selected_model}")

# ================================
# KPI
# ================================
col1, col2 = st.columns(2)

with col1:
    st.metric("Matches", len(df))

with col2:
    if "correct" in df.columns:
        st.metric("Accuracy", f"{df['correct'].mean()*100:.1f}%")

# ================================
# TABS
# ================================
tab_labels = ["Data"]

if len(models) >= 2:
    tab_labels.append("Model Comparison")

tabs = st.tabs(tab_labels)

# ================================
# TAB 1
# ================================
with tabs[0]:
    st.dataframe(df)

# ================================
# TAB 2: MODEL COMPARISON
# ================================
if len(models) >= 2:
    with tabs[1]:
        st.subheader("Model Comparison")

        rows = []
        for name, mdf in models.items():
            row = {
                "Model": name,
                "Matches": len(mdf)
            }

            if "correct" in mdf.columns:
                row["Accuracy"] = round(mdf["correct"].mean()*100, 2)

            rows.append(row)

        summary = pd.DataFrame(rows)
        st.dataframe(summary)
