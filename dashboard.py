import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="La Liga Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ================================
# COLORS (same as compare.py)
# ================================
BLUE   = "#58A6FF"
RED    = "#FF5252"
GREEN  = "#00C9A7"
GOLD   = "#FFC300"

# ================================
# LOAD CSV
# ================================
def load_df(file):
    df = pd.read_csv(file, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

# ================================
# FIXED PREPROCESSING (IMPORTANT)
# ================================
def prepare_df(df):
    df = df.copy().reset_index(drop=True)

    # ✅ FIX 1: FORCE MATCH ALIGNMENT
    df["Match"] = np.arange(1, len(df) + 1)

    # ================= SCORE =================
    if "Predict Score" in df.columns and "Actual Score" in df.columns:
        def split_score(s):
            try:
                h, a = str(s).split("-")
                return int(h), int(a)
            except:
                return np.nan, np.nan

        df[["pred_h","pred_a"]] = df["Predict Score"].apply(lambda x: pd.Series(split_score(x)))
        df[["act_h","act_a"]]   = df["Actual Score"].apply(lambda x: pd.Series(split_score(x)))

        df["pred_total"] = df["pred_h"] + df["pred_a"]
        df["act_total"]  = df["act_h"] + df["act_a"]

        df["exact_match"]   = df["Predict Score"] == df["Actual Score"]
        df["residual"]      = df["act_total"] - df["pred_total"]

    # ================= OUTCOME =================
    if "Predict Outcome" in df.columns and "Actual Outcome" in df.columns:
        df["correct"] = df["Predict Outcome"] == df["Actual Outcome"]

    # ================= TOP-N =================
    if "1 from top 5" in df.columns:
        df["top5"] = df["1 from top 5"].str.upper() == "YES"
    if "1 from top 3" in df.columns:
        df["top3"] = df["1 from top 3"].str.upper() == "YES"
    if "1 from top 2" in df.columns:
        df["top2"] = df["1 from top 2"].str.upper() == "YES"

    # ================= STRICT COLUMN FIX =================
    # ✅ FIX 2: NO AUTO DETECTION (matches python code exactly)
    if "xG" in df.columns:
        df["xg_col"] = df["xG"].astype(float)

    if "Goals" in df.columns:
        df["goal_col"] = df["Goals"].astype(float)

    if "Win Prob" in df.columns:
        df["WIN PROBABILITY"] = df["Win Prob"].astype(float)

    if "Draw Prob" in df.columns:
        df["DRAW PROBABILITY"] = df["Draw Prob"].astype(float)

    if "Loss Prob" in df.columns:
        df["LOSS PROBABILITY"] = df["Loss Prob"].astype(float)

    return df

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.title("⚽ La Liga Hub")

    uploaded_files = st.file_uploader(
        "Upload CSVs",
        type="csv",
        accept_multiple_files=True
    )

models = {}
if uploaded_files:
    for f in uploaded_files:
        df = load_df(f)
        df = prepare_df(df)
        models[f.name] = df

if not models:
    st.warning("Upload CSV to start")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
df = models[selected_model]
match_no = df["Match"]

# ================================
# HEADER
# ================================
st.title("⚽ La Liga Prediction Dashboard")

# ================================
# KPI FIXED
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Matches", len(df))

with col2:
    if "correct" in df:
        valid = df["correct"].dropna()
        acc = (valid.sum() / len(valid)) * 100 if len(valid) > 0 else 0
        st.metric("Accuracy", f"{acc:.1f}%")

with col3:
    if "xg_col" in df:
        st.metric("Avg xG", f"{df['xg_col'].mean():.2f}")

with col4:
    if "xg_col" in df and "goal_col" in df:
        mae = (df["goal_col"] - df["xg_col"]).abs().mean()
        st.metric("MAE", f"{mae:.2f}")

# ================================
# GRAPH 1 (xG vs Goals)
# ================================
if "xg_col" in df and "goal_col" in df:
    fig, ax = plt.subplots()

    ax.scatter(df["xg_col"], df["goal_col"], color=BLUE)
    ax.plot([0, df["xg_col"].max()], [0, df["xg_col"].max()], "w--")

    ax.set_title("xG vs Goals")

    st.pyplot(fig)

# ================================
# GRAPH 2 (Residual)
# ================================
if "xg_col" in df and "goal_col" in df:
    residual = df["goal_col"] - df["xg_col"]

    fig, ax = plt.subplots()
    ax.scatter(df["xg_col"], residual, color=BLUE)
    ax.axhline(0, color="white", linestyle="--")

    ax.set_title("Residual Plot")

    st.pyplot(fig)

# ================================
# MODEL COMPARISON (FIXED)
# ================================
if len(models) >= 2:
    st.header("Model Comparison")

    rows = []
    for name, mdf in models.items():

        valid = mdf["correct"].dropna() if "correct" in mdf else []

        row = {
            "Model": name,
            "Matches": len(mdf),
            "Accuracy": (valid.sum()/len(valid))*100 if len(valid) > 0 else 0
        }

        if "xg_col" in mdf and "goal_col" in mdf:
            row["MAE"] = (mdf["goal_col"] - mdf["xg_col"]).abs().mean()

        rows.append(row)

    summary = pd.DataFrame(rows)

    st.dataframe(summary)

    # BAR CHART
    fig, ax = plt.subplots()

    ax.bar(summary["Model"], summary["Accuracy"], color=GREEN)

    ax.set_title("Model Accuracy Comparison")

    st.pyplot(fig)
