# ================================
# FIXED MODELS LOADING (ONLY CHANGE)
# ================================

models = {}   # {label: df}

if uploaded_files:
    for i, f in enumerate(uploaded_files):
        raw = load_df(f)
        df  = prepare_df(raw)

        # ✅ FIX: unique model name (prevents overwrite)
        clean_name = f.name.replace(".csv", "")
        unique_name = f"{clean_name} ({i+1})"

        models[unique_name] = df
