from __future__ import annotations

import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from typing import Tuple

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Prediksi Kepuasan Penumpang",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS (simple, clean, modern)
# ============================================================
CUSTOM_CSS = """
<style>
/* Global spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Hero */
.hero {
  background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(16,185,129,0.20));
  border: 1px solid rgba(255,255,255,0.08);
  padding: 18px 18px;
  border-radius: 18px;
}
.hero h1 { margin: 0; font-size: 28px; }
.hero p { margin: 6px 0 0 0; opacity: 0.9; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  padding: 14px 14px;
  border-radius: 16px;
}
.kpi {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.kpi .pill {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  padding: 10px 12px;
  border-radius: 999px;
  font-size: 13px;
}

/* Section title */
.section-title {
  margin-top: 0.2rem;
  font-size: 18px;
  font-weight: 700;
}

/* Small helper text */
.muted { opacity: 0.85; font-size: 13px; }

/* Better expander */
.streamlit-expanderHeader { font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# PATH MODEL (robust for deploy)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

BEST_MODEL_PATH = OUTPUTS_DIR / "best_model.joblib"
LOGREG_MODEL_PATH = OUTPUTS_DIR / "logreg_model.joblib"  # opsional
XGB_MODEL_PATH = OUTPUTS_DIR / "xgb_model.joblib"        # opsional

TARGET_COL = "satisfaction"
DROP_COLS = ["Unnamed: 0"]
DROP_ID_COLS = ["id", "ID"]  # rekomendasi: jangan ikut diprediksi

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model(path: str | Path):
    return joblib.load(path)

@st.cache_data
def read_csv_cached(file):
    return pd.read_csv(file)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def get_feature_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c != TARGET_COL]
    cols = [c for c in cols if c not in DROP_ID_COLS]
    return cols

def predict_one(model, X_row: pd.DataFrame):
    proba = float(model.predict_proba(X_row)[:, 1][0])
    label = "satisfied" if proba >= 0.5 else "neutral or dissatisfied"
    return proba, label

def label_id_to_display(lbl: str) -> str:
    return "puas" if lbl == "satisfied" else "netral atau tidak puas"

def set_form_from_row(df: pd.DataFrame, feature_cols: list, row_idx: int):
    row = df.iloc[row_idx]
    for col in feature_cols:
        key = f"inp_{col}"
        val = row[col]
        if not is_numeric_series(df[col]):
            val = "" if pd.isna(val) else str(val)
        st.session_state[key] = val

def validate_columns_against_model(model, X_input: pd.DataFrame) -> Tuple[bool, str]:
    try:
        _ = model.predict_proba(X_input)
        return True, ""
    except Exception as e:
        return False, str(e)

def fmt_feature_name(col: str) -> str:
    return (
        col.replace("_", " ")
        .replace("  ", " ")
        .strip()
        .title()
    )

# ============================================================
# Header / Hero
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>✈️ PREDIKSI KEPUASAN PENUMPANG MASKAPAI PENERBANGAN</h1>
  <p class="muted">
    Upload <b>1 dataset CSV</b> → pilih baris (auto-fill) → edit jika perlu → klik <b>Prediksi</b>.
    <br/>Kolom <b>id/ID</b> otomatis diabaikan agar konsisten dengan model.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# Sidebar: Upload & Model
# ============================================================
st.sidebar.markdown("## ⚙️ Kontrol")
st.sidebar.markdown("### 1) Unggah Dataset (CSV)")
uploaded = st.sidebar.file_uploader("Unggah dataset CSV", type=["csv"])

st.sidebar.markdown("### 2) Model")

if not BEST_MODEL_PATH.exists():
    st.sidebar.error("❌ best_model.joblib tidak ditemukan.")
    st.sidebar.info("Pastikan file ada di folder outputs/ dan sudah ikut ter-push ke repo.")
    st.stop()

# Load hanya best_model saat startup (wajib)
best_model = load_model(BEST_MODEL_PATH)
st.sidebar.success("✅ best_model.joblib dimuat")

# Jangan load LR & XGB saat startup (lebih ringan). Akan di-load saat Prediksi ditekan.
has_lr_file = LOGREG_MODEL_PATH.exists()
has_xgb_file = XGB_MODEL_PATH.exists()

if not has_lr_file or not has_xgb_file:
    st.sidebar.info("Perbandingan LR vs XGB aktif jika file model LR dan XGB tersedia.")
else:
    st.sidebar.success("✅ File LR & XGB terdeteksi (akan di-load saat prediksi)")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tips")
st.sidebar.caption("• Jika prediksi error, kemungkinan kolom dataset tidak sama dengan saat training.")
st.sidebar.caption("• Gunakan train.csv/test.csv yang berasal dari dataset yang sama.")

# ============================================================
# If no upload -> stop
# ============================================================
if uploaded is None:
    st.info("Silakan upload 1 file CSV (train.csv atau test.csv).")
    st.stop()

# Read & clean
df = clean_df(read_csv_cached(uploaded))
has_target = TARGET_COL in df.columns
feature_cols = get_feature_cols(df)

# ============================================================
# Top KPI strip
# ============================================================
k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
with k1:
    st.metric("Jumlah baris", df.shape[0])
with k2:
    st.metric("Jumlah kolom", df.shape[1])
with k3:
    st.metric("Ada target?", "Ya" if has_target else "Tidak")
with k4:
    st.markdown(
        f"""
        <div class="card">
          <div class="section-title">Ringkas</div>
          <div class="kpi">
            <div class="pill">Fitur digunakan: <b>{len(feature_cols)}</b></div>
            <div class="pill">Target: <b>{TARGET_COL}</b></div>
            <div class="pill">Ambang: <b>0.5</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ============================================================
# Tabs layout: Dataset / Input+Prediksi (MENYATU)
# ============================================================
tab_ds, tab_main = st.tabs(["📄 Dataset", "🧾 Input & Prediksi"])

# ------------------------------------------------------------
# TAB 1: Dataset
# ------------------------------------------------------------
with tab_ds:
    st.markdown("### 📄 Pratinjau Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("🔎 Info kolom & tipe data"):
        info_df = pd.DataFrame({
            "kolom": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: Input + Prediksi (MENYATU)
# ------------------------------------------------------------
with tab_main:
    st.markdown("### 🎯 Pilih Baris (Auto-fill)")

    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0

    def on_row_change():
        set_form_from_row(df, feature_cols, st.session_state.row_idx)

    row_idx = st.selectbox(
        "Pilih indeks baris",
        options=list(range(df.shape[0])),
        key="row_idx",
        on_change=on_row_change
    )

    # init form once
    if "form_initialized" not in st.session_state:
        set_form_from_row(df, feature_cols, row_idx)
        st.session_state.form_initialized = True

    selected_row = df.iloc[[row_idx]].copy()

    cA, cB = st.columns([3, 2])
    with cA:
        with st.expander("🔎 Lihat baris yang dipilih (mentah)"):
            st.dataframe(selected_row, use_container_width=True)

    with cB:
        if st.button("♻️ Reset form sesuai baris pilihan", use_container_width=True):
            set_form_from_row(df, feature_cols, row_idx)
            st.success("Form direset sesuai baris terpilih.")

    st.markdown("---")
    st.markdown("### ✍️ Input Manual (Terisi otomatis, bisa diedit)")

    left, right = st.columns(2)
    half = (len(feature_cols) + 1) // 2
    left_cols = feature_cols[:half]
    right_cols = feature_cols[half:]

    def render_cols(cols, container):
        with container:
            for col in cols:
                s = df[col]
                label = fmt_feature_name(col)

                if is_numeric_series(s):
                    finite_vals = pd.to_numeric(s, errors="coerce").dropna()
                    vmin = float(finite_vals.min()) if len(finite_vals) else 0.0
                    vmax = float(finite_vals.max()) if len(finite_vals) else 1.0
                    step = (vmax - vmin) / 100.0 if vmax > vmin else 1.0
                    step = float(step) if step > 0 else 1.0

                    st.number_input(
                        label,
                        min_value=vmin,
                        max_value=vmax,
                        step=step,
                        key=f"inp_{col}",
                    )
                else:
                    options = sorted(list(pd.Series(s).dropna().astype(str).unique()))
                    if not options:
                        options = [""]

                    cur = st.session_state.get(f"inp_{col}", "")
                    if cur not in options:
                        st.session_state[f"inp_{col}"] = options[0]

                    st.selectbox(
                        label,
                        options=options,
                        key=f"inp_{col}",
                    )

    render_cols(left_cols, left)
    render_cols(right_cols, right)

    # Build X_input dari session_state
    user_values = {col: st.session_state.get(f"inp_{col}") for col in feature_cols}
    X_input = pd.DataFrame([user_values], columns=feature_cols)

    st.markdown("### Ringkasan input yang akan diprediksi")
    st.dataframe(X_input, use_container_width=True)

    # ========================================================
    # PREDIKSI (langsung di bawah form)
    # ========================================================
    st.markdown("---")
    st.markdown("### 🧠 Prediksi & Perbandingan")

    run = st.button("🚀 Prediksi", type="primary", use_container_width=True)

    if run:
        ok, err = validate_columns_against_model(best_model, X_input)
        if not ok:
            st.error("❌ Dataset / kolom input tidak cocok dengan model yang dilatih.")
            st.code(err)
            st.stop()

        best_proba, best_label = predict_one(best_model, X_input)

        # Result cards
        left_r, right_r = st.columns([2, 1])
        with left_r:
            st.markdown(
                f"""
                <div class="card">
                  <div class="section-title">Hasil Prediksi (Model Terbaik)</div>
                  <p class="muted">Label prediksi berdasarkan ambang 0.5</p>
                  <h2 style="margin:0.2rem 0 0.2rem 0;">{label_id_to_display(best_label).upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_r:
            st.markdown(
                f"""
                <div class="card">
                  <div class="section-title">Probabilitas Puas</div>
                  <p class="muted">P(satisfied)</p>
                  <h2 style="margin:0.2rem 0 0.2rem 0;">{best_proba:.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.progress(min(max(best_proba, 0.0), 1.0))

        # Ground truth (if available)
        if has_target:
            gt = str(selected_row[TARGET_COL].iloc[0]).strip()
            gt_id = "satisfied" if gt.lower() == "satisfied" else "neutral or dissatisfied"

            st.markdown("---")
            st.markdown("### ✅ Ground Truth (Label Asli dari dataset)")
            st.info(f"Label asli baris pilihan: **{label_id_to_display(gt_id)}**")

            if gt_id == best_label:
                st.success("✅ Prediksi BENAR (sesuai label asli).")
            else:
                st.error("❌ Prediksi SALAH (tidak sesuai label asli).")

        # ====================================================
        # Model comparison (LR vs XGB) - lazy load here
        # ====================================================
        st.markdown("---")
        st.markdown("### 🔁 Perbandingan Model (Jika tersedia)")
        cL, cR = st.columns(2)

        def render_model_box(container, title, model):
            with container:
                st.markdown(f"#### {title}")
                if model is None:
                    st.warning(f"Belum ada file model untuk {title}.")
                    return

                ok_m, err_m = validate_columns_against_model(model, X_input)
                if not ok_m:
                    st.warning(f"Model {title} tidak cocok dengan kolom input.")
                    st.code(err_m)
                    return

                proba_m, label_m = predict_one(model, X_input)
                st.write(f"**Label:** {label_id_to_display(label_m)}")
                st.write(f"**Proba puas:** {proba_m:.4f}")
                st.progress(min(max(proba_m, 0.0), 1.0))

        # Lazy-load only when needed (button pressed)
        logreg_model = load_model(LOGREG_MODEL_PATH) if has_lr_file else None
        xgb_model = load_model(XGB_MODEL_PATH) if has_xgb_file else None

        render_model_box(cL, "Regresi Logistik", logreg_model)
        render_model_box(cR, "XGBoost", xgb_model)

        st.caption("Catatan: ambang batas 0.5 untuk menentukan label dari probabilitas.")
