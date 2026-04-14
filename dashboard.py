# ================================
# FIXED MODEL LOADING SECTION
# ================================

models = {}   # {label: df}

if uploaded_files:
    for i, f in enumerate(uploaded_files):
        raw = load_df(f)
        df  = prepare_df(raw)

        # ✅ FIX: unique names (no overwrite)
        clean_name = f.name.replace(".csv", "")
        unique_name = f"{clean_name} ({i+1})"

        models[unique_name] = df

# ================================
# SAFE HANDLING (NO FILE)
# ================================

if not models:
    st.warning("Upload at least one CSV file")
    st.stop()

# ================================
# MODEL SELECTOR
# ================================

with st.sidebar:
    selected_model = st.selectbox("🔍 Active Model", list(models.keys()))

# ✅ FIX: prevent crash
if selected_model is None:
    st.stop()

df = models[selected_model]
