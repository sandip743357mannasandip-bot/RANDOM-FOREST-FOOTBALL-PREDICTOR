import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- PAGE ----------------
st.set_page_config(page_title="La Liga Dashboard", layout="wide")

# ---------------- COLORS ----------------
BLUE="#58A6FF"; RED="#FF5252"; GREEN="#00C9A7"; GOLD="#FFC300"

# ---------------- SHOW FIG ----------------
def show(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

# ---------------- LOAD ----------------
def load_df(file):
    df = pd.read_csv(file, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

# ---------------- 🔥 FIXED PREPARE ----------------
def prepare_df(df):
    df = df.copy()
    lc = {c.lower(): c for c in df.columns}

    # SCORE
    if "predict score" in lc and "actual score" in lc:
        ps, as_ = lc["predict score"], lc["actual score"]

        def parse_score(s):
            try:
                h, a = str(s).split("-")
                return int(h), int(a)
            except:
                return np.nan, np.nan

        df[["pred_h","pred_a"]] = pd.DataFrame(df[ps].apply(parse_score).tolist())
        df[["act_h","act_a"]] = pd.DataFrame(df[as_].apply(parse_score).tolist())

        df["pred_total"] = df["pred_h"] + df["pred_a"]
        df["act_total"]  = df["act_h"]  + df["act_a"]
        df["exact_match"] = df[ps].astype(str) == df[as_].astype(str)
        df["residual"] = df["act_total"] - df["pred_total"]

    # OUTCOME
    if "predict outcome" in lc and "actual outcome" in lc:
        po, ao = lc["predict outcome"], lc["actual outcome"]
        df["predict outcome"] = df[po].astype(str)
        df["actual outcome"]  = df[ao].astype(str)
        df["correct"] = df["predict outcome"] == df["actual outcome"]

    # 🔥 IMPORTANT FIX
    if "match" in lc:
        df["Match"] = pd.to_numeric(df[lc["match"]], errors="coerce")
    else:
        df["Match"] = np.arange(1, len(df)+1)

    return df

# ---------------- SIDEBAR ----------------
st.sidebar.title("Upload CSV")
file = st.sidebar.file_uploader("Upload file", type="csv")

if not file:
    st.title("⚽ Upload CSV to start")
    st.stop()

df = prepare_df(load_df(file))

# ---------------- HEADER ----------------
st.title("⚽ La Liga Dashboard")

col1,col2 = st.columns(2)
col1.metric("Matches", len(df))
if "correct" in df.columns:
    col2.metric("Accuracy", f"{df['correct'].mean()*100:.1f}%")

# ---------------- TABS ----------------
tabs = st.tabs(["Score Analysis","Outcome"])

# ---------------- SCORE TAB ----------------
with tabs[0]:
    if "pred_total" in df.columns:

        fig, ax = plt.subplots()
        ax.scatter(df["pred_total"], df["act_total"], color=BLUE)
        ax.plot([0,6],[0,6],"w--")
        ax.set_title("Pred vs Actual Goals")
        show(fig)

        # 🔥 SAFE LINE (NO CRASH)
        exact_idx = df.loc[df["exact_match"], "Match"]
        exact_val = df.loc[df["exact_match"], "act_total"]

        fig, ax = plt.subplots()
        ax.plot(df["Match"], df["act_total"], label="Actual")
        ax.plot(df["Match"], df["pred_total"], label="Pred")
        ax.scatter(exact_idx, exact_val, color=GREEN)
        ax.legend()
        show(fig)

# ---------------- OUTCOME TAB ----------------
with tabs[1]:
    if "correct" in df.columns:
        st.write("Correct:", df["correct"].sum())
        st.write("Wrong:", len(df)-df["correct"].sum())
