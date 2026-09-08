"""
app.py
------
Streamlit web application for the Social Media Screen-Time & Mental-Health
Wellbeing Band predictor.

Run locally:  streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wellbeing Band Predictor",
    page_icon="🧠",
    layout="wide",
)

# ── Load model artifact ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "model.pkl")

@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_artifact()

# ── Helper: band colour ───────────────────────────────────────────────────────
BAND_CONFIG = {
    "Good":     {"emoji": "🟢", "colour": "#22c55e", "msg": "Your predicted wellbeing is **Good**. Keep up the healthy habits!"},
    "Moderate": {"emoji": "🟡", "colour": "#f59e0b", "msg": "Your predicted wellbeing is **Moderate**. Small lifestyle tweaks may help."},
    "At-risk":  {"emoji": "🔴", "colour": "#ef4444", "msg": "Your predicted wellbeing is **At-risk**. Consider reaching out for support."},
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Social Media & Mental-Health Wellbeing Predictor")
st.markdown(
    "This tool uses a machine-learning model trained on the **2026 Social Media "
    "Screen-Time & Mental Health** dataset to predict your **wellbeing band**: "
    "_Good_, _Moderate_, or _At-risk_."
)
st.divider()

# ── Guard: model not trained yet ──────────────────────────────────────────────
if artifact is None:
    st.error(
        "⚠️ No trained model found at `models/model.pkl`.\n\n"
        "Run `python train_model.py` first, then reload this page."
    )
    st.stop()

pipeline         = artifact["pipeline"]
le               = artifact["label_encoder"]
numeric_cols     = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]
cat_options      = artifact["cat_options"]
num_stats        = artifact["num_stats"]
best_name        = artifact["best_model_name"]
cv_results       = artifact["cv_results"]

# ── Sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.metric("Best Model", best_name)

    st.subheader("Cross-Validation Accuracy (5-fold)")
    cv_data = {
        "Model": list(cv_results.keys()),
        "Mean Acc": [f"{np.mean(v):.4f}" for v in cv_results.values()],
        "Std":      [f"{np.std(v):.4f}"  for v in cv_results.values()],
    }
    st.dataframe(pd.DataFrame(cv_data).set_index("Model"), use_container_width=True)

    st.subheader("Target Classes")
    for cls in le.classes_:
        cfg = BAND_CONFIG.get(cls, {})
        st.markdown(f"{cfg.get('emoji','')} **{cls}**")

    st.divider()
    st.caption("Dataset: Social Media Screentime & Mental Health 2026 · 7,000 participants")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Your Details")
st.markdown("Fill in the fields below and click **Predict Wellbeing Band**.")

def _num_default(col, stat="50%"):
    val = num_stats.get(col, {}).get(stat, 0.0)
    return float(val) if val is not None else 0.0

def _num_min(col):
    return float(num_stats.get(col, {}).get("min", 0.0))

def _num_max(col):
    return float(num_stats.get(col, {}).get("max", 9999.0))

col1, col2, col3 = st.columns(3)

input_values = {}

# ── Demographic & usage ───────────────────────────────────────────────────────
with col1:
    st.markdown("**👤 Demographics**")
    input_values["age"] = st.number_input(
        "Age", min_value=10, max_value=100,
        value=int(_num_default("age")), step=1
    )
    input_values["gender"] = st.selectbox(
        "Gender", options=cat_options.get("gender", ["Male", "Female", "Non-binary"])
    )
    input_values["occupation"] = st.selectbox(
        "Occupation", options=cat_options.get("occupation", [])
    )
    input_values["region"] = st.selectbox(
        "Region", options=cat_options.get("region", [])
    )

    st.markdown("**📱 Platform Usage**")
    input_values["most_used_platform"] = st.selectbox(
        "Most-Used Platform", options=cat_options.get("most_used_platform", [])
    )
    input_values["platforms_used_count"] = st.number_input(
        "Number of Platforms Used", min_value=1, max_value=20,
        value=int(_num_default("platforms_used_count")), step=1
    )
    input_values["primary_purpose"] = st.selectbox(
        "Primary Purpose of Social Media",
        options=cat_options.get("primary_purpose", [])
    )

with col2:
    st.markdown("**⏱️ Screen Time & Habits**")
    input_values["daily_screen_hours"] = st.slider(
        "Daily Screen Hours", min_value=0.0, max_value=24.0,
        value=float(_num_default("daily_screen_hours")), step=0.1
    )
    input_values["daily_notifications"] = st.number_input(
        "Daily Notifications Received", min_value=0, max_value=1000,
        value=int(_num_default("daily_notifications")), step=1
    )
    input_values["night_time_use"] = st.selectbox(
        "Night-time Use Frequency",
        options=cat_options.get("night_time_use",
                                ["Never", "Sometimes", "Often", "Every night"])
    )
    input_values["minutes_to_first_check_after_waking"] = st.number_input(
        "Minutes to First Check After Waking", min_value=0, max_value=300,
        value=int(_num_default("minutes_to_first_check_after_waking")), step=1
    )
    input_values["avg_sleep_hours"] = st.slider(
        "Average Sleep Hours", min_value=0.0, max_value=14.0,
        value=float(_num_default("avg_sleep_hours")), step=0.1
    )
    input_values["physical_activity_days_per_week"] = st.number_input(
        "Physical Activity Days/Week", min_value=0, max_value=7,
        value=int(_num_default("physical_activity_days_per_week")), step=1
    )

    st.markdown("**🔧 Behaviours**")
    input_values["uses_screen_time_limits"] = st.selectbox(
        "Uses Screen-Time Limits?",
        options=cat_options.get("uses_screen_time_limits", ["Yes", "No"])
    )
    input_values["attempted_digital_detox"] = st.selectbox(
        "Attempted Digital Detox?",
        options=cat_options.get("attempted_digital_detox",
                                ["No", "Yes, failed", "Yes, succeeded"])
    )
    input_values["seeks_mental_health_support"] = st.selectbox(
        "Seeks Mental-Health Support?",
        options=cat_options.get("seeks_mental_health_support",
                                ["No", "Yes", "Considering it"])
    )

with col3:
    st.markdown("**🧠 Mental-Health Scores**")
    input_values["anxiety_score_0to27"] = st.slider(
        "Anxiety Score (0–27)", min_value=0, max_value=27,
        value=int(_num_default("anxiety_score_0to27")), step=1
    )
    input_values["low_mood_score_0to27"] = st.slider(
        "Low-Mood Score (0–27)", min_value=0, max_value=27,
        value=int(_num_default("low_mood_score_0to27")), step=1
    )
    input_values["life_satisfaction_1to10"] = st.slider(
        "Life Satisfaction (1–10)", min_value=1, max_value=10,
        value=int(_num_default("life_satisfaction_1to10")), step=1
    )
    input_values["loneliness_1to10"] = st.slider(
        "Loneliness (1–10)", min_value=1, max_value=10,
        value=int(_num_default("loneliness_1to10")), step=1
    )
    input_values["self_esteem_1to10"] = st.slider(
        "Self-Esteem (1–10)", min_value=1, max_value=10,
        value=int(_num_default("self_esteem_1to10")), step=1
    )
    input_values["fomo_1to10"] = st.slider(
        "FOMO (Fear of Missing Out) (1–10)", min_value=1, max_value=10,
        value=int(_num_default("fomo_1to10")), step=1
    )
    input_values["social_comparison_1to10"] = st.slider(
        "Social Comparison (1–10)", min_value=1, max_value=10,
        value=int(_num_default("social_comparison_1to10")), step=1
    )

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Wellbeing Band", type="primary", use_container_width=True):
    # Build DataFrame in the exact column order expected by the pipeline
    feature_names = artifact["feature_names"]
    input_df = pd.DataFrame([input_values])[feature_names]

    # Predict class and probabilities
    pred_enc       = pipeline.predict(input_df)[0]
    pred_label     = le.inverse_transform([pred_enc])[0]
    pred_proba     = pipeline.predict_proba(input_df)[0]   # shape: (n_classes,)
    classes_order  = le.inverse_transform(range(len(le.classes_)))

    cfg = BAND_CONFIG.get(pred_label, {"emoji": "⚪", "colour": "#6b7280", "msg": ""})

    st.markdown("### 🎯 Prediction Result")
    st.markdown(
        f"<div style='background:{cfg['colour']}22; border-left:6px solid {cfg['colour']};"
        f"padding:1rem 1.5rem; border-radius:6px; font-size:1.15rem;'>"
        f"{cfg['emoji']} {cfg['msg']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Prediction Confidence")
    prob_df = pd.DataFrame(
        {"Wellbeing Band": classes_order, "Probability": pred_proba}
    ).set_index("Wellbeing Band")

    for band, row in prob_df.iterrows():
        b_cfg  = BAND_CONFIG.get(band, {})
        colour = b_cfg.get("colour", "#3b82f6")
        pct    = row["Probability"] * 100
        st.markdown(
            f"**{b_cfg.get('emoji','')} {band}** — {pct:.1f}%"
        )
        st.progress(float(row["Probability"]))

    with st.expander("🔍 Input Summary (features sent to model)"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: trained on Social Media Screentime & Mental Health 2026 dataset · "
    f"Best classifier: **{best_name}** · "
    "Predictions are for informational purposes only and are not medical advice."
)
