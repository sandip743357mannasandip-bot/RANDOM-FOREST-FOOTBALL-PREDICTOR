"""
La Liga Prediction Dashboard  —  Streamlit
Replicates every plot from compare.py exactly (same colours, same logic).
Upload one or more AFTER_PREDICTION / COMPLETE_PREDICTION CSV files and
switch between them; a Model Comparison tab is added automatically when
two or more files are loaded.

Run:
    pip install streamlit pandas matplotlib numpy scikit-learn
    streamlit run dashboard.py
"""

import io
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.metrics import confusion_matrix

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="La Liga Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  EXACT SAME COLOURS AS compare.py
# ══════════════════════════════════════════════════════════════════
BLUE   = "#58A6FF"
RED    = "#FF5252"
GREEN  = "#00C9A7"
GOLD   = "#FFC300"
WIN_C  = "#00C9A7"
DRAW_C = "#FFC300"
LOSS_C = "#FF5252"
HIT    = "#00C9A7"
MISS   = "#FF5252"

STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "grid.color":       "#21262D",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.size":        11,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": "#30363D",
}
plt.rcParams.update(STYLE)

# ── helper: render fig in streamlit ──────────────────────────────
def show(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════
#  CUSTOM CSS  (dark sidebar + dark main)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
body, .stApp { background-color: #0D1117; color: white; }
section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
.stTabs [data-baseweb="tab-list"]  { background: #161B22; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"]       { background: #0D1117; color: #8B949E; border-radius: 6px; padding: 6px 18px; font-size: 0.82rem; letter-spacing: 1px; }
.stTabs [aria-selected="true"]     { background: #21262D !important; color: #FFC300 !important; }
div[data-testid="metric-container"]{ background: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 14px 18px; }
div[data-testid="metric-container"] label { color: #8B949E !important; font-size: 0.7rem; letter-spacing: 1.5px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 1.8rem; }
hr { border-color: #30363D; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — upload multiple models
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ La Liga Hub")
    st.markdown("---")
    st.markdown("### 📂 Upload Model CSVs")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload AFTER_PREDICTION.csv or COMPLETE_PREDICTION.csv files"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8B949E'>Each file = one model.<br>"
        "Columns auto-detected.<br>"
        "Upload 2+ files to unlock<br><b style='color:#FFC300'>Model Comparison</b> tab.</small>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════
#  LOAD & CLEAN CSVs
# ══════════════════════════════════════════════════════════════════
def load_df(file):
    df = pd.read_csv(file, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

def prepare_df(df):
    """Add all derived columns used by compare.py."""
    df = df.copy()
    # lowercase lookup helpers
    lc = {c.lower(): c for c in df.columns}

    # ── score columns ──
    if "predict score" in lc and "actual score" in lc:
        ps, as_ = lc["predict score"], lc["actual score"]
        def parse_score(s):
            try:
                h, a = str(s).strip().split("-")
                return int(h), int(a)
            except:
                return None, None
        df[["pred_h","pred_a"]] = pd.DataFrame(df[ps].apply(parse_score).tolist(), index=df.index)
        df[["act_h", "act_a"]]  = pd.DataFrame(df[as_].apply(parse_score).tolist(),  index=df.index)
        df["pred_total"]  = df["pred_h"] + df["pred_a"]
        df["act_total"]   = df["act_h"]  + df["act_a"]
        df["exact_match"] = df[ps].str.strip() == df[as_].str.strip()
        df["residual"]    = df["act_total"] - df["pred_total"]

    # ── outcome columns ──
    if "predict outcome" in lc and "actual outcome" in lc:
        po, ao = lc["predict outcome"], lc["actual outcome"]
        df["predict outcome"] = df[po].str.strip()
        df["actual outcome"]  = df[ao].str.strip()
        df["outcome_match"]   = df["predict outcome"] == df["actual outcome"]
        df["correct"]         = df["outcome_match"]

    # ── top-N ──
    for col_key, new_col in [("1 from top 5","top5"),("1 from top 3","top3"),("1 from top 2","top2")]:
        if col_key in lc:
            df[new_col] = df[lc[col_key]].str.strip().str.upper() == "YES"

    # ── xg / goals ──
    for c in df.columns:
        cl = c.lower()
        if "xg" in cl:
            df["xg_col"] = df[c].astype(float)
        if "goal" in cl and "xg" not in cl:
            df["goal_col"] = df[c].astype(float)

    # ── probability ──
    for c in df.columns:
        cl = c.lower()
        if "win prob" in cl:   df["WIN PROBABILITY"]  = df[c].astype(float)
        if "draw prob" in cl:  df["DRAW PROBABILITY"] = df[c].astype(float)
        if "loss prob" in cl:  df["LOSS PROBABILITY"] = df[c].astype(float)

    # ── match number ──
    if "match" in lc:
        df["Match"] = pd.to_numeric(df[lc["match"]], errors="coerce")

    return df

models = {}   # {label: df}
if uploaded_files:
    for f in uploaded_files:
        raw = load_df(f)
        df  = prepare_df(raw)
        models[f.name] = df

# ══════════════════════════════════════════════════════════════════
#  LANDING — no file uploaded
# ══════════════════════════════════════════════════════════════════
if not models:
    st.markdown("""
    <div style='text-align:center; padding:80px 20px'>
        <div style='font-size:5rem'>⚽</div>
        <h1 style='color:white; font-size:2.2rem; letter-spacing:3px'>LA LIGA PREDICTION DASHBOARD</h1>
        <p style='color:#8B949E; font-size:1rem; margin-top:12px'>
            Upload your CSV file(s) in the sidebar to begin.<br>
            Every chart from your Python <code>compare.py</code> is reproduced exactly.
        </p>
        <div style='margin-top:40px; display:flex; justify-content:center; gap:24px; flex-wrap:wrap'>
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px 28px;min-width:180px'>
                <div style='font-size:2rem'>📊</div><div style='color:#8B949E;margin-top:8px;font-size:.85rem'>Section 1<br><b style="color:white">xG Analysis</b></div>
            </div>
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px 28px;min-width:180px'>
                <div style='font-size:2rem'>⚽</div><div style='color:#8B949E;margin-top:8px;font-size:.85rem'>Section 2<br><b style="color:white">Score Prediction</b></div>
            </div>
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px 28px;min-width:180px'>
                <div style='font-size:2rem'>🎯</div><div style='color:#8B949E;margin-top:8px;font-size:.85rem'>Section 3<br><b style="color:white">Outcome Prediction</b></div>
            </div>
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px 28px;min-width:180px'>
                <div style='font-size:2rem'>🏆</div><div style='color:#8B949E;margin-top:8px;font-size:.85rem'>Section 4<br><b style="color:white">Top-N Hit Rate</b></div>
            </div>
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;padding:20px 28px;min-width:180px'>
                <div style='font-size:2rem'>📈</div><div style='color:#8B949E;margin-top:8px;font-size:.85rem'>Section 5<br><b style="color:white">Probabilities</b></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  MODEL SELECTOR (sidebar)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    selected_model = st.selectbox("🔍 Active Model", list(models.keys()))

df = models[selected_model]
match_no = df["Match"].values if "Match" in df.columns else np.arange(1, len(df)+1)

# ══════════════════════════════════════════════════════════════════
#  HEADER + KPIs
# ══════════════════════════════════════════════════════════════════
st.markdown(f"## ⚽ La Liga Prediction Dashboard")
st.markdown(f"<small style='color:#8B949E'>Model: **{selected_model}** &nbsp;|&nbsp; {len(df)} matches</small>", unsafe_allow_html=True)
st.markdown("---")

# KPI row
k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
with k1:
    st.metric("Matches", len(df))
with k2:
    if "correct" in df.columns:
        acc = df["correct"].mean()*100
        st.metric("Outcome Acc", f"{acc:.1f}%")
with k3:
    if "xg_col" in df.columns:
        st.metric("Avg xG", f"{df['xg_col'].mean():.2f}")
with k4:
    if "xg_col" in df.columns and "goal_col" in df.columns:
        mae = (df["goal_col"] - df["xg_col"]).abs().mean()
        st.metric("MAE (xG)", f"{mae:.3f}")
with k5:
    if "top5" in df.columns:
        st.metric("Top-5 Hit%", f"{df['top5'].mean()*100:.1f}%")
with k6:
    if "top3" in df.columns:
        st.metric("Top-3 Hit%", f"{df['top3'].mean()*100:.1f}%")
with k7:
    if "top2" in df.columns:
        st.metric("Top-2 Hit%", f"{df['top2'].mean()*100:.1f}%")
with k8:
    if "WIN PROBABILITY" in df.columns:
        st.metric("Avg Win Prob", f"{df['WIN PROBABILITY'].mean():.1f}%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
#  BUILD TABS
# ══════════════════════════════════════════════════════════════════
tab_labels = ["📊 xG Analysis", "⚽ Score Prediction",
              "🎯 Outcome Prediction", "🏆 Top-N Hit Rate",
              "📈 Probabilities", "📋 Data Table"]
if len(models) >= 2:
    tab_labels.append("🔀 Model Comparison")

tabs = st.tabs(tab_labels)

# ══════════════════════════════════════════════════════════════════
#  TAB 1 — SECTION 1  xG vs Actual Goals
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    if "xg_col" not in df.columns or "goal_col" not in df.columns:
        st.warning("No xG / Goals columns detected in this CSV.")
    else:
        xg   = df["xg_col"].values
        goal = df["goal_col"].values

        # ── Scatter ──────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.scatter(xg, goal, color=BLUE, edgecolors="white", lw=0.5)
            ax.plot([0, xg.max()], [0, xg.max()], "w--", lw=1.5, label="Perfect")
            ax.set_xlabel("Predicted xG"); ax.set_ylabel("Actual Goals")
            ax.set_title("xG vs Actual Goals"); ax.legend()
            fig.tight_layout(); show(fig)

        # ── Residual scatter ─────────────────────────────────────
        with c2:
            residual = goal - xg
            fig, ax = plt.subplots()
            ax.scatter(xg, residual, color=BLUE, edgecolors="white", lw=0.5)
            ax.axhline(0, color="white", lw=1.5, ls="--")
            ax.set_xlabel("Predicted xG"); ax.set_ylabel("Error (Goals − xG)")
            ax.set_title("Residual Plot"); ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

        # ── Distribution ─────────────────────────────────────────
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots()
            ax.hist(xg,   bins=20, alpha=0.6, color=RED,  label="Predicted xG",  edgecolor="white", lw=0.5)
            ax.hist(goal, bins=20, alpha=0.6, color=BLUE, label="Actual Goals",   edgecolor="white", lw=0.5)
            ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
            ax.set_title("Distribution: xG vs Goals"); ax.legend()
            fig.tight_layout(); show(fig)

        # ── Line ─────────────────────────────────────────────────
        with c4:
            fig, ax = plt.subplots()
            ax.plot(xg,   color=RED,  lw=2, ls="--", label="Predicted xG")
            ax.plot(goal, color=BLUE, lw=2,           label="Actual Goals")
            ax.set_xlabel("Match Index"); ax.set_ylabel("Value")
            ax.set_title("Match-wise Comparison"); ax.legend()
            fig.tight_layout(); show(fig)

# ══════════════════════════════════════════════════════════════════
#  TAB 2 — SECTION 2  Score Prediction
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    need = ["pred_total","act_total","exact_match","outcome_match","residual"]
    if not all(c in df.columns for c in need):
        st.warning("Score columns (predict score / actual score) not found.")
    else:
        colors_scatter = [GREEN if e else (GOLD if o else RED)
                          for e, o in zip(df["exact_match"], df["outcome_match"])]
        lim = max(df["pred_total"].max(), df["act_total"].max()) + 0.5

        # ── Scatter: total goals ──────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(9,7))
            ax.scatter(df["pred_total"], df["act_total"],
                       c=colors_scatter, s=100, edgecolors="white", linewidths=0.6, zorder=3)
            ax.plot([0,lim],[0,lim],"w--",lw=1.5,alpha=0.6,label="Perfect prediction")
            ax.set_xlim(-0.3,lim); ax.set_ylim(-0.3,lim)
            ax.set_xlabel("Predicted Total Goals"); ax.set_ylabel("Actual Total Goals")
            ax.set_title("Predicted Score vs Actual Score  –  Total Goals", fontweight="bold")
            ax.grid(True)
            ax.legend(handles=[
                mpatches.Patch(color=GREEN, label="Exact score match"),
                mpatches.Patch(color=GOLD,  label="Outcome correct only"),
                mpatches.Patch(color=RED,   label="Wrong outcome"),
                plt.Line2D([0],[0],color="white",ls="--",label="Perfect prediction"),
            ], fontsize=9, framealpha=0.15)
            fig.tight_layout(); show(fig)

        # ── Residual bar ─────────────────────────────────────────
        with c2:
            bar_colors = [GREEN if r >= 0 else RED for r in df["residual"]]
            fig, ax = plt.subplots(figsize=(14,5))
            ax.bar(match_no, df["residual"], color=bar_colors, edgecolor="#0D1117", lw=0.4, alpha=0.9)
            ax.axhline(0, color="white", lw=1.5, ls="--")
            ax.axhline(df["residual"].mean(), color=GOLD, lw=1.5, ls=":",
                       label=f"Mean error = {df['residual'].mean():.2f}")
            ax.set_xlabel("Match Number"); ax.set_ylabel("Residual  (Actual − Predicted goals)")
            ax.set_title("Residual Plot  –  Predict Score vs Actual Score", fontweight="bold")
            ax.set_xticks(match_no); ax.grid(True, axis="y")
            ax.legend(fontsize=10, framealpha=0.15)
            fig.tight_layout(); show(fig)

        # ── Distribution: total goals ─────────────────────────────
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(9,6))
            bins = np.arange(-0.5, 9.5, 1)
            ax.hist(df["pred_total"], bins=bins, alpha=0.7, color=RED,  label="Predicted Total Goals", edgecolor="white", lw=0.5)
            ax.hist(df["act_total"],  bins=bins, alpha=0.7, color=BLUE, label="Actual Total Goals",    edgecolor="white", lw=0.5)
            ax.set_xlabel("Total Goals in Match"); ax.set_ylabel("Number of Matches")
            ax.set_title("Distribution  –  Predicted vs Actual Total Goals", fontweight="bold")
            ax.legend(fontsize=10, framealpha=0.15); ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

        # ── Line: match-by-match ──────────────────────────────────
        with c4:
            fig, ax = plt.subplots(figsize=(14,5))
            ax.plot(match_no, df["pred_total"], color=RED,  lw=2, ls="--", marker="s", ms=6, label="Predicted Score Total")
            ax.plot(match_no, df["act_total"],  color=BLUE, lw=2,           marker="o", ms=6, label="Actual Score Total")
            ax.fill_between(match_no, df["pred_total"], df["act_total"],
                            where=(df["act_total"] >= df["pred_total"]), alpha=0.12, color=BLUE, label="Under-predicted")
            ax.fill_between(match_no, df["pred_total"], df["act_total"],
                            where=(df["act_total"] <  df["pred_total"]), alpha=0.12, color=RED,  label="Over-predicted")
            exact_idx = df[df["exact_match"]]["Match"]
            exact_val = df[df["exact_match"]]["act_total"]
            ax.scatter(exact_idx, exact_val, color=GREEN, s=120, zorder=5, edgecolors="white", lw=0.8, label="Exact score hit")
            ax.set_xlabel("Match Number"); ax.set_ylabel("Total Goals")
            ax.set_title("Match-by-Match  –  Predicted Score vs Actual Score", fontweight="bold")
            ax.set_xticks(match_no); ax.legend(fontsize=9, framealpha=0.15); ax.grid(True)
            fig.tight_layout(); show(fig)

        # ── Side-by-side bar ──────────────────────────────────────
        st.markdown("#### Side-by-Side: Predicted vs Actual Total Goals")
        bw = 0.38
        fig, ax = plt.subplots(figsize=(16,5))
        ax.bar(match_no - bw/2, df["pred_total"], bw, color=RED,  alpha=0.85, label="Predicted Total", edgecolor="#0D1117", lw=0.4)
        ax.bar(match_no + bw/2, df["act_total"],  bw, color=BLUE, alpha=0.85, label="Actual Total",    edgecolor="#0D1117", lw=0.4)
        for _, row in df[df["exact_match"]].iterrows():
            ax.bar(row["Match"] - bw/2, row["pred_total"], bw, color=GREEN, alpha=0.9, edgecolor="#0D1117")
            ax.bar(row["Match"] + bw/2, row["act_total"],  bw, color=GREEN, alpha=0.9, edgecolor="#0D1117")
            ax.text(row["Match"], max(row["pred_total"], row["act_total"]) + 0.1,
                    "HIT", ha="center", va="bottom", color=GREEN, fontsize=7, fontweight="bold")
        ax.set_xlabel("Match Number"); ax.set_ylabel("Total Goals")
        ax.set_title("Side-by-Side  –  Predicted vs Actual Score  (Green = Exact Hit)", fontweight="bold")
        ax.set_xticks(match_no); ax.legend(fontsize=10, framealpha=0.15); ax.grid(True, axis="y")
        fig.tight_layout(); show(fig)

        # ── Donut: accuracy ───────────────────────────────────────
        c5, _ = st.columns([1, 1])
        with c5:
            exact_n   = int(df["exact_match"].sum())
            outcome_n = int(df["outcome_match"].sum()) - exact_n
            wrong_n   = len(df) - int(df["outcome_match"].sum())
            fig, ax = plt.subplots(figsize=(7,7))
            wedges, texts, autotexts = ax.pie(
                [exact_n, outcome_n, wrong_n],
                labels=[f"Exact Score\n({exact_n})", f"Right Outcome\n({outcome_n})", f"Wrong\n({wrong_n})"],
                colors=[GREEN, GOLD, RED], autopct="%1.1f%%", startangle=90, pctdistance=0.78,
                wedgeprops={"edgecolor": "#0D1117", "linewidth": 2.5},
            )
            for at in autotexts: at.set_fontsize(10); at.set_color("white")
            for t  in texts:     t.set_fontsize(11); t.set_color("white")
            ax.add_artist(plt.Circle((0,0), 0.55, color="#161B22"))
            ax.text(0, 0, f"{df['outcome_match'].mean()*100:.0f}%\nOutcome\nAccuracy",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            ax.set_title(f"Prediction Outcome Accuracy  –  {len(df)} Matches", fontweight="bold", pad=15)
            fig.tight_layout(); show(fig)

# ══════════════════════════════════════════════════════════════════
#  TAB 3 — SECTION 3  Outcome Prediction
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    if "predict outcome" not in df.columns or "actual outcome" not in df.columns:
        st.warning("Outcome columns not found.")
    else:
        outcome_color = {"W": WIN_C, "D": DRAW_C, "L": LOSS_C}
        outcome_num   = {"W": 2, "D": 1, "L": 0}
        outcomes      = ["W", "D", "L"]
        pred_num      = df["predict outcome"].map(outcome_num)
        actual_num    = df["actual outcome"].map(outcome_num)

        # ── Strip chart ───────────────────────────────────────────
        st.markdown("#### Match-by-Match: Predicted vs Actual Outcome")
        fig, ax = plt.subplots(figsize=(16,5))
        ax.scatter(match_no, [1.1]*len(df),
                   c=[outcome_color.get(o, GOLD) for o in df["predict outcome"]],
                   s=200, marker="s", edgecolors="white", lw=0.5, zorder=3)
        ax.scatter(match_no, [0.9]*len(df),
                   c=[outcome_color.get(o, GOLD) for o in df["actual outcome"]],
                   s=200, marker="o", edgecolors="white", lw=0.5, zorder=3)
        for m, correct in zip(match_no, df["correct"]):
            ax.plot([m,m],[0.93,1.07], color=WIN_C if correct else LOSS_C, lw=2, alpha=0.7, zorder=2)
        ax.set_yticks([0.9, 1.1]); ax.set_yticklabels(["Actual", "Predicted"], fontsize=12)
        ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(0.3, len(df)+0.7); ax.set_ylim(0.75, 1.25)
        ax.set_title("Match-by-Match  –  Predicted vs Actual Outcome", fontweight="bold")
        ax.set_xlabel("Match Number"); ax.grid(True, axis="x")
        ax.legend(handles=[
            mpatches.Patch(color=WIN_C,  label="Win (W)"),
            mpatches.Patch(color=DRAW_C, label="Draw (D)"),
            mpatches.Patch(color=LOSS_C, label="Loss (L)"),
            plt.Line2D([0],[0], color=WIN_C,  lw=2, label="Correct prediction"),
            plt.Line2D([0],[0], color=LOSS_C, lw=2, label="Wrong prediction"),
        ], fontsize=9, framealpha=0.15, loc="upper right")
        fig.tight_layout(); show(fig)

        # ── Confusion matrix + frequency bar ─────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            from pandas import crosstab
            matrix = crosstab(df["predict outcome"], df["actual outcome"]).reindex(index=outcomes, columns=outcomes, fill_value=0)
            fig, ax = plt.subplots(figsize=(7,6))
            im = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0)
            for i in range(len(matrix)):
                for j in range(len(matrix.columns)):
                    val = matrix.values[i,j]
                    color = "white" if val > matrix.values.max()*0.6 else BLUE
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=20, fontweight="bold", color=color)
            ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
            ax.set_xticklabels(["Actual W","Actual D","Actual L"], fontsize=12)
            ax.set_yticklabels(["Pred W","Pred D","Pred L"], fontsize=12)
            ax.set_title("Confusion Matrix  –  Predicted vs Actual Outcome", fontweight="bold", pad=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(); show(fig)

        with c2:
            st.markdown("#### Outcome Frequency: Predicted vs Actual")
            pred_counts   = df["predict outcome"].value_counts().reindex(outcomes, fill_value=0)
            actual_counts = df["actual outcome"].value_counts().reindex(outcomes, fill_value=0)
            bw = 0.35; x = np.arange(3)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(x - bw/2, pred_counts,   bw, color=LOSS_C, alpha=0.85, label="Predicted", edgecolor="#0D1117", lw=0.5)
            ax.bar(x + bw/2, actual_counts, bw, color=BLUE,   alpha=0.85, label="Actual",    edgecolor="#0D1117", lw=0.5)
            for xi, (p, a) in enumerate(zip(pred_counts, actual_counts)):
                ax.text(xi-bw/2, p+0.3, str(p), ha="center", fontsize=12, fontweight="bold", color=LOSS_C)
                ax.text(xi+bw/2, a+0.3, str(a), ha="center", fontsize=12, fontweight="bold", color=BLUE)
            ax.set_xticks(x); ax.set_xticklabels(["Win (W)","Draw (D)","Loss (L)"], fontsize=12)
            ax.set_ylabel("Number of Matches")
            ax.set_title("Outcome Frequency  –  Predicted vs Actual", fontweight="bold")
            ax.legend(fontsize=11, framealpha=0.15); ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

        # ── Outcome line ──────────────────────────────────────────
        st.markdown("#### Outcome Line: Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(match_no, pred_num,   color=LOSS_C, lw=2, ls="--", marker="s", ms=7, label="Predicted Outcome")
        ax.plot(match_no, actual_num, color=BLUE,   lw=2,           marker="o", ms=7, label="Actual Outcome")
        ax.fill_between(match_no, pred_num, actual_num,
                        where=df["correct"].values,  alpha=0.12, color=WIN_C,  label="Correct prediction")
        ax.fill_between(match_no, pred_num, actual_num,
                        where=~df["correct"].values, alpha=0.12, color=LOSS_C, label="Wrong prediction")
        ax.set_yticks([0,1,2]); ax.set_yticklabels(["Loss (L)","Draw (D)","Win (W)"], fontsize=11)
        ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8); ax.set_xlabel("Match Number")
        ax.set_title("Match-by-Match Outcome  –  Predicted vs Actual", fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.15); ax.grid(True)
        fig.tight_layout(); show(fig)

        # ── Outcome donut ─────────────────────────────────────────
        c3, _ = st.columns([1,1])
        with c3:
            correct_n = int(df["correct"].sum()); wrong_n = len(df) - correct_n
            fig, ax = plt.subplots(figsize=(7,7))
            wedges, texts, autotexts = ax.pie(
                [correct_n, wrong_n],
                labels=[f"Correct\n({correct_n})", f"Wrong\n({wrong_n})"],
                colors=[WIN_C, LOSS_C], autopct="%1.1f%%", startangle=90, pctdistance=0.78,
                wedgeprops={"edgecolor":"#0D1117","linewidth":2.5},
            )
            for at in autotexts: at.set_fontsize(13); at.set_color("white")
            for t  in texts:     t.set_fontsize(13); t.set_color("white")
            ax.add_artist(plt.Circle((0,0), 0.55, color="#161B22"))
            ax.text(0, 0, f"{correct_n/len(df)*100:.0f}%\nAccuracy",
                    ha="center", va="center", fontsize=16, fontweight="bold", color="white")
            ax.set_title(f"Overall Outcome Prediction Accuracy  –  {len(df)} Matches", fontweight="bold", pad=15)
            fig.tight_layout(); show(fig)

# ══════════════════════════════════════════════════════════════════
#  TAB 4 — SECTION 4  Top-N Hit Rate
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    if not all(c in df.columns for c in ["top5","top3","top2"]):
        st.warning("Top-N columns (1 FROM TOP 5/3/2) not found.")
    else:
        total  = len(df)
        t5_yes = int(df["top5"].sum()); t5_no = total - t5_yes
        t3_yes = int(df["top3"].sum()); t3_no = total - t3_yes
        t2_yes = int(df["top2"].sum()); t2_no = total - t2_yes
        tiers  = ["Top 5", "Top 3", "Top 2"]

        # ── Strip chart ───────────────────────────────────────────
        st.markdown("#### Match-by-Match: Did Actual Score Appear in Top N?")
        fig, ax = plt.subplots(figsize=(18,5))
        for y, data, label in zip([3,2,1], [df["top5"],df["top3"],df["top2"]], tiers):
            ax.scatter(match_no, [y]*total,
                       c=[HIT if v else MISS for v in data],
                       s=220, marker="s", edgecolors="white", lw=0.5, zorder=3)
            ax.text(-0.5, y, label, ha="right", va="center", fontsize=12, fontweight="bold", color="white")
        ax.set_yticks([]); ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(-1, total+1); ax.set_ylim(0.4, 3.6); ax.set_xlabel("Match Number")
        ax.set_title("Match-by-Match  –  Did Actual Score Appear in Top N Predictions?", fontweight="bold")
        ax.grid(True, axis="x")
        ax.legend(handles=[
            mpatches.Patch(color=HIT,  label="YES – in prediction list"),
            mpatches.Patch(color=MISS, label="NO  – not in prediction list"),
        ], fontsize=10, framealpha=0.15, loc="upper right")
        fig.tight_layout(); show(fig)

        # ── Bar chart ─────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            x = np.arange(3); bw = 0.35
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(x-bw/2, [t5_yes,t3_yes,t2_yes], bw, color=HIT,  alpha=0.9, label="YES (Hit)",  edgecolor="#0D1117", lw=0.5)
            ax.bar(x+bw/2, [t5_no, t3_no, t2_no],  bw, color=MISS, alpha=0.9, label="NO (Miss)",  edgecolor="#0D1117", lw=0.5)
            for xi,(y,n) in enumerate(zip([t5_yes,t3_yes,t2_yes],[t5_no,t3_no,t2_no])):
                ax.text(xi-bw/2, y+0.3, str(y), ha="center", fontsize=13, fontweight="bold", color=HIT)
                ax.text(xi+bw/2, n+0.3, str(n), ha="center", fontsize=13, fontweight="bold", color=MISS)
            ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=13); ax.set_ylabel("Number of Matches")
            ax.set_title(f"Hit Count per Prediction Tier  –  {total} Matches", fontweight="bold")
            ax.legend(fontsize=11, framealpha=0.15); ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

        # ── Cumulative ────────────────────────────────────────────
        with c2:
            fig, ax = plt.subplots(figsize=(14,5))
            ax.plot(match_no, df["top5"].cumsum().values, color=BLUE, lw=2.5, marker="o", ms=5, label="Top 5")
            ax.plot(match_no, df["top3"].cumsum().values, color=GOLD, lw=2.5, marker="s", ms=5, label="Top 3")
            ax.plot(match_no, df["top2"].cumsum().values, color=HIT,  lw=2.5, marker="^", ms=5, label="Top 2")
            ax.plot(match_no, match_no, color="white", lw=1, ls="--", alpha=0.3, label="Perfect (100%)")
            ax.set_xlabel("Match Number"); ax.set_ylabel("Cumulative Hits")
            ax.set_title("Cumulative Hits over the Season  –  Top 2 / 3 / 5", fontweight="bold")
            ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8)
            ax.legend(fontsize=10, framealpha=0.15); ax.grid(True)
            fig.tight_layout(); show(fig)

        # ── Stacked horizontal bar ────────────────────────────────
        st.markdown("#### Hit Rate per Prediction Tier")
        hit_pcts = [t5_yes/total*100, t3_yes/total*100, t2_yes/total*100]
        fig, ax = plt.subplots(figsize=(9,4))
        for yp, hp, label in zip([2,1,0], hit_pcts, tiers):
            mp = 100 - hp
            ax.barh(yp, hp,       color=HIT,  alpha=0.9, edgecolor="#0D1117", lw=0.5)
            ax.barh(yp, mp, left=hp, color=MISS, alpha=0.9, edgecolor="#0D1117", lw=0.5)
            ax.text(hp/2,      yp, f"{hp:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
            ax.text(hp+mp/2,   yp, f"{mp:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
        ax.set_yticks([2,1,0]); ax.set_yticklabels(tiers, fontsize=13)
        ax.set_xticks(np.arange(0,101,10))
        ax.set_xlabel("Percentage (%)"); ax.set_title("Hit Rate per Prediction Tier", fontweight="bold")
        ax.axvline(50, color="white", lw=1, ls="--", alpha=0.4)
        ax.legend(handles=[
            mpatches.Patch(color=HIT,  label="YES – Hit %"),
            mpatches.Patch(color=MISS, label="NO – Miss %"),
        ], fontsize=10, framealpha=0.15)
        ax.grid(True, axis="x")
        fig.tight_layout(); show(fig)

        # ── Three donuts ──────────────────────────────────────────
        st.markdown("#### Hit Rate Donuts – Did Actual Score Appear in Top N?")
        fig, axes = plt.subplots(1, 3, figsize=(14,6))
        for ax, label, yes, no in zip(axes, tiers,
                                      [t5_yes,t3_yes,t2_yes],
                                      [t5_no, t3_no, t2_no]):
            wedges, texts, autotexts = ax.pie(
                [yes, no], labels=[f"YES\n({yes})", f"NO\n({no})"], colors=[HIT, MISS],
                autopct="%1.1f%%", startangle=90, pctdistance=0.78,
                wedgeprops={"edgecolor":"#0D1117","linewidth":2.5},
            )
            for at in autotexts: at.set_fontsize(11); at.set_color("white")
            for t  in texts:     t.set_fontsize(11); t.set_color("white")
            ax.add_artist(plt.Circle((0,0), 0.55, color="#161B22"))
            ax.text(0, 0, f"{yes/total*100:.0f}%\nHit Rate",
                    ha="center", va="center", fontsize=13, fontweight="bold", color="white")
            ax.set_title(f"Prediction  {label}", fontweight="bold", pad=12, fontsize=13)
        fig.suptitle("Hit Rate – Did Actual Score Appear in Top N?", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout(); show(fig)

# ══════════════════════════════════════════════════════════════════
#  TAB 5 — SECTION 5  Probabilities
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    prob_cols = ["WIN PROBABILITY","DRAW PROBABILITY","LOSS PROBABILITY"]
    if not all(c in df.columns for c in prob_cols) or "correct" not in df.columns:
        st.warning("Probability columns or outcome columns not found.")
    else:
        WIN_P  = df["WIN PROBABILITY"]
        DRAW_P = df["DRAW PROBABILITY"]
        LOSS_P = df["LOSS PROBABILITY"]

        # ── Stacked area ──────────────────────────────────────────
        st.markdown("#### Win / Draw / Loss Probability per Match – Actual Outcome as Dots")
        fig, ax = plt.subplots(figsize=(16,6))
        ax.stackplot(match_no, WIN_P, DRAW_P, LOSS_P,
                     labels=["Win Probability","Draw Probability","Loss Probability"],
                     colors=[WIN_C, DRAW_C, LOSS_C], alpha=0.75)
        for _, row in df.iterrows():
            oc = row["actual outcome"]
            y_pos = {"W":95,"D":50,"L":5}.get(oc, 50)
            ax.scatter(row["Match"], y_pos,
                       color={"W":WIN_C,"D":DRAW_C,"L":LOSS_C}.get(oc, GOLD),
                       s=80, zorder=5, edgecolors="white", lw=0.8)
        ax.axhline(50, color="white", lw=1, ls="--", alpha=0.3)
        ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Probability (%)"); ax.set_xlabel("Match Number")
        ax.set_title("Win / Draw / Loss Probability per Match  –  Actual Outcome as Dots", fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(color=WIN_C,  label="Win Probability"),
            mpatches.Patch(color=DRAW_C, label="Draw Probability"),
            mpatches.Patch(color=LOSS_C, label="Loss Probability"),
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white", ms=7, label="Actual outcome dot"),
        ], fontsize=9, framealpha=0.15, loc="upper right")
        ax.grid(True, axis="x")
        fig.tight_layout(); show(fig)

        # ── Scatter panels ────────────────────────────────────────
        st.markdown("#### Assigned Probability vs Whether That Outcome Actually Occurred")
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        for ax, (prob, color, outcome_key, title) in zip(axes, [
            (WIN_P,  WIN_C,  "W", "Win Probability"),
            (DRAW_P, DRAW_C, "D", "Draw Probability"),
            (LOSS_P, LOSS_C, "L", "Loss Probability"),
        ]):
            mask = df["actual outcome"] == outcome_key
            ax.scatter(prob[mask],  df.loc[mask,  "Match"], color=color,    s=100, edgecolors="white", lw=0.6, zorder=3, label=f"Actual {outcome_key}")
            ax.scatter(prob[~mask], df.loc[~mask, "Match"], color="#30363D", s=80,  edgecolors="white", lw=0.4, zorder=2, label=f"Actual ≠ {outcome_key}")
            ax.axvline(50, color="white", lw=1, ls="--", alpha=0.4, label="50% mark")
            ax.set_xlabel("Probability (%)"); ax.set_ylabel("Match Number")
            ax.set_title(title, fontweight="bold", color=color)
            ax.legend(fontsize=8, framealpha=0.15); ax.grid(True)
        fig.suptitle("Assigned Probability vs Whether That Outcome Actually Occurred",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout(); show(fig)

        # ── Probability line with markers ─────────────────────────
        st.markdown("#### Probability Lines – Correct (★) vs Wrong (✗)")
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(match_no, WIN_P,  color=WIN_C,  lw=2,   marker="o", ms=5, label="Win Prob")
        ax.plot(match_no, DRAW_P, color=DRAW_C, lw=1.5, marker="s", ms=4, label="Draw Prob", ls="--")
        ax.plot(match_no, LOSS_P, color=LOSS_C, lw=1.5, marker="^", ms=4, label="Loss Prob", ls=":")
        ax.axhline(50, color="white", lw=1, ls="--", alpha=0.3, label="50% line")
        ax.scatter(df[df["correct"]]["Match"],  WIN_P[df["correct"]],
                   color="white", s=150, marker="*", zorder=5, label="Correct prediction")
        ax.scatter(df[~df["correct"]]["Match"], WIN_P[~df["correct"]],
                   color=LOSS_C,  s=120, marker="X", zorder=5, label="Wrong prediction")
        ax.set_xticks(match_no); ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Probability (%)"); ax.set_xlabel("Match Number")
        ax.set_title("Probability Lines  –  Correct (★) vs Wrong (✗) Predictions Marked", fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.15, loc="upper right"); ax.grid(True)
        fig.tight_layout(); show(fig)

        # ── Avg probability by actual outcome ─────────────────────
        c1, c2 = st.columns(2)
        with c1:
            groups = ["W","D","L"]
            avg_win_p  = [df[df["actual outcome"]==g]["WIN PROBABILITY"].mean()  for g in groups]
            avg_draw_p = [df[df["actual outcome"]==g]["DRAW PROBABILITY"].mean() for g in groups]
            avg_loss_p = [df[df["actual outcome"]==g]["LOSS PROBABILITY"].mean() for g in groups]
            x = np.arange(3); bw = 0.26
            fig, ax = plt.subplots(figsize=(9,6))
            b1 = ax.bar(x-bw,   avg_win_p,  bw, color=WIN_C,  alpha=0.85, label="Avg Win Prob",  edgecolor="#0D1117", lw=0.5)
            b2 = ax.bar(x,      avg_draw_p, bw, color=DRAW_C, alpha=0.85, label="Avg Draw Prob", edgecolor="#0D1117", lw=0.5)
            b3 = ax.bar(x+bw,   avg_loss_p, bw, color=LOSS_C, alpha=0.85, label="Avg Loss Prob", edgecolor="#0D1117", lw=0.5)
            for bars in [b1, b2, b3]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(x); ax.set_xticklabels(["Actual Win","Actual Draw","Actual Loss"], fontsize=12)
            ax.set_ylabel("Average Probability (%)"); ax.legend(fontsize=10, framealpha=0.15)
            ax.set_title("Average Probabilities Assigned  –  Grouped by Actual Outcome", fontweight="bold")
            ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

        # ── Distribution: win prob correct vs wrong ────────────────
        with c2:
            fig, ax = plt.subplots(figsize=(9,6))
            bins = np.arange(20, 85, 5)
            ax.hist(WIN_P[df["correct"]],  bins=bins, alpha=0.75, color=WIN_C,  edgecolor="white", lw=0.5,
                    label=f"Correct predictions ({df['correct'].sum()})")
            ax.hist(WIN_P[~df["correct"]], bins=bins, alpha=0.75, color=LOSS_C, edgecolor="white", lw=0.5,
                    label=f"Wrong predictions ({(~df['correct']).sum()})")
            ax.axvline(WIN_P[df["correct"]].mean(),  color=WIN_C,  lw=2, ls="--",
                       label=f"Correct mean = {WIN_P[df['correct']].mean():.1f}%")
            ax.axvline(WIN_P[~df["correct"]].mean(), color=LOSS_C, lw=2, ls="--",
                       label=f"Wrong mean = {WIN_P[~df['correct']].mean():.1f}%")
            ax.set_xlabel("Win Probability (%)"); ax.set_ylabel("Number of Matches")
            ax.set_title("Win Probability Distribution  –  Correct vs Wrong Predictions", fontweight="bold")
            ax.legend(fontsize=10, framealpha=0.15); ax.grid(True, axis="y")
            fig.tight_layout(); show(fig)

# ══════════════════════════════════════════════════════════════════
#  TAB 6 — DATA TABLE
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("#### Full Match Data")
    st.dataframe(df, use_container_width=True, height=600)

# ══════════════════════════════════════════════════════════════════
#  TAB 7 — MODEL COMPARISON  (only when 2+ models)
# ══════════════════════════════════════════════════════════════════
if len(models) >= 2:
    with tabs[6]:
        st.markdown("## 🔀 Model Comparison")
        model_names = list(models.keys())

        # ── Build summary table ───────────────────────────────────
        rows = []
        for name, mdf in models.items():
            row = {"Model": name, "Matches": len(mdf)}
            if "correct" in mdf.columns:
                row["Outcome Acc %"] = round(mdf["correct"].mean()*100, 1)
            if "xg_col" in mdf.columns and "goal_col" in mdf.columns:
                row["MAE (xG)"] = round((mdf["goal_col"]-mdf["xg_col"]).abs().mean(), 3)
                row["Avg xG"]   = round(mdf["xg_col"].mean(), 2)
            if "top5" in mdf.columns:
                row["Top-5 Hit%"] = round(mdf["top5"].mean()*100, 1)
                row["Top-3 Hit%"] = round(mdf["top3"].mean()*100, 1)
                row["Top-2 Hit%"] = round(mdf["top2"].mean()*100, 1)
            if "WIN PROBABILITY" in mdf.columns:
                row["Avg Win Prob"] = round(mdf["WIN PROBABILITY"].mean(), 1)
            rows.append(row)

        summary = pd.DataFrame(rows).set_index("Model")
        st.dataframe(summary.style.highlight_max(axis=0, color="#1a3a2a").highlight_min(axis=0, color="#3a1a1a"),
                     use_container_width=True)

        st.markdown("---")

        # ── Outcome accuracy bar ──────────────────────────────────
        if "Outcome Acc %" in summary.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Outcome Accuracy per Model")
                fig, ax = plt.subplots(figsize=(8, 5))
                bar_colors = [GREEN, BLUE, GOLD, RED, "#B57BFF"][:len(summary)]
                bars = ax.bar(range(len(summary)), summary["Outcome Acc %"],
                              color=bar_colors[:len(summary)], edgecolor="#0D1117", lw=0.5, alpha=0.9)
                for bar, val in zip(bars, summary["Outcome Acc %"]):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
                ax.set_xticks(range(len(summary)))
                ax.set_xticklabels([n[:20] for n in summary.index], rotation=15, ha="right")
                ax.set_ylabel("Outcome Accuracy (%)"); ax.grid(True, axis="y")
                ax.set_title("Outcome Accuracy – Model Comparison", fontweight="bold")
                fig.tight_layout(); show(fig)

            # ── Top-N hit rate comparison ─────────────────────────
            if "Top-5 Hit%" in summary.columns:
                with c2:
                    st.markdown("#### Top-N Hit Rate Comparison")
                    x = np.arange(3); bw = 0.8 / len(summary)
                    fig, ax = plt.subplots(figsize=(9,5))
                    palette = [GREEN, BLUE, GOLD, RED, "#B57BFF"]
                    for i, (name, row) in enumerate(summary.iterrows()):
                        vals = [row.get("Top-5 Hit%",0), row.get("Top-3 Hit%",0), row.get("Top-2 Hit%",0)]
                        offset = (i - len(summary)/2 + 0.5) * bw
                        ax.bar(x + offset, vals, bw*0.9, color=palette[i % len(palette)],
                               alpha=0.9, label=name[:18], edgecolor="#0D1117", lw=0.4)
                    ax.set_xticks(x); ax.set_xticklabels(["Top 5","Top 3","Top 2"], fontsize=12)
                    ax.set_ylabel("Hit Rate (%)"); ax.legend(fontsize=9, framealpha=0.15)
                    ax.set_title("Top-N Hit Rate – Model Comparison", fontweight="bold")
                    ax.grid(True, axis="y")
                    fig.tight_layout(); show(fig)

        # ── MAE comparison ────────────────────────────────────────
        if "MAE (xG)" in summary.columns:
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### xG MAE Comparison (lower = better)")
                fig, ax = plt.subplots(figsize=(8,5))
                palette = [GREEN, BLUE, GOLD, RED, "#B57BFF"]
                bars = ax.bar(range(len(summary)), summary["MAE (xG)"],
                              color=palette[:len(summary)], edgecolor="#0D1117", lw=0.5, alpha=0.9)
                for bar, val in zip(bars, summary["MAE (xG)"]):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
                ax.set_xticks(range(len(summary)))
                ax.set_xticklabels([n[:20] for n in summary.index], rotation=15, ha="right")
                ax.set_ylabel("Mean Absolute Error"); ax.grid(True, axis="y")
                ax.set_title("xG MAE – Model Comparison", fontweight="bold")
                fig.tight_layout(); show(fig)

            # ── Avg xG comparison ─────────────────────────────────
            with c4:
                st.markdown("#### Average xG per Match")
                fig, ax = plt.subplots(figsize=(8,5))
                bars = ax.bar(range(len(summary)), summary["Avg xG"],
                              color=palette[:len(summary)], edgecolor="#0D1117", lw=0.5, alpha=0.9)
                for bar, val in zip(bars, summary["Avg xG"]):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                            f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
                ax.set_xticks(range(len(summary)))
                ax.set_xticklabels([n[:20] for n in summary.index], rotation=15, ha="right")
                ax.set_ylabel("Avg Predicted xG"); ax.grid(True, axis="y")
                ax.set_title("Average xG – Model Comparison", fontweight="bold")
                fig.tight_layout(); show(fig)

        # ── Cumulative accuracy overlay ───────────────────────────
        if all("correct" in models[n].columns for n in model_names):
            st.markdown("#### Cumulative Outcome Accuracy Over Season – All Models Overlaid")
            palette = [GREEN, BLUE, GOLD, RED, "#B57BFF"]
            fig, ax = plt.subplots(figsize=(16,6))
            for i, (name, mdf) in enumerate(models.items()):
                mn = mdf["Match"].values if "Match" in mdf.columns else np.arange(1,len(mdf)+1)
                cum_acc = [(mdf["correct"].iloc[:j+1].mean()*100) for j in range(len(mdf))]
                ax.plot(mn, cum_acc, color=palette[i % len(palette)], lw=2.5,
                        marker="o", ms=4, label=name[:25])
            ax.axhline(50, color="white", lw=1, ls="--", alpha=0.3, label="50% baseline")
            ax.set_xlabel("Match Number"); ax.set_ylabel("Cumulative Accuracy (%)")
            ax.set_title("Cumulative Outcome Accuracy – All Models", fontweight="bold")
            ax.legend(fontsize=9, framealpha=0.15); ax.grid(True)
            fig.tight_layout(); show(fig)

        # ── Win probability line overlay ──────────────────────────
        if all("WIN PROBABILITY" in models[n].columns for n in model_names):
            st.markdown("#### Win Probability per Match – All Models Overlaid")
            palette = [GREEN, BLUE, GOLD, RED, "#B57BFF"]
            fig, ax = plt.subplots(figsize=(16,6))
            for i, (name, mdf) in enumerate(models.items()):
                mn = mdf["Match"].values if "Match" in mdf.columns else np.arange(1,len(mdf)+1)
                ax.plot(mn, mdf["WIN PROBABILITY"], color=palette[i % len(palette)],
                        lw=2, marker="o", ms=4, label=name[:25])
            ax.axhline(50, color="white", lw=1, ls="--", alpha=0.3)
            ax.set_xlabel("Match Number"); ax.set_ylabel("Win Probability (%)")
            ax.set_title("Win Probability per Match – All Models Overlaid", fontweight="bold")
            ax.legend(fontsize=9, framealpha=0.15); ax.grid(True)
            fig.tight_layout(); show(fig)
