import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load model, scaler, and explainer ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# === Define model features and their slider ranges ===
model_features = [
    'area_worst',
    'concavity_mean',
    'area_mean',
    'concavity_worst',
    'area_se',
    'smoothness_worst',
    'texture_mean',
    'concavity_se',
    'symmetry_worst',
    'smoothness_mean',
    'fractal_dimension_worst'
]

slider_ranges = {
    "area_worst": (185.2, 4254.0),
    "concavity_mean": (0.0, 0.4268),
    "area_mean": (143.5, 2501.0),
    "concavity_worst": (0.0, 1.252),
    "area_se": (6.802, 542.2),
    "smoothness_worst": (0.07117, 0.2226),
    "texture_mean": (9.71, 39.28),
    "concavity_se": (0.0, 0.396),
    "symmetry_worst": (0.1565, 1.058),
    "smoothness_mean": (0.05263, 0.1634),
    "fractal_dimension_worst": (0.05504, 0.2075),
}

# === Layout: sliders on the left, output on the right ===
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("## 🛠️ Input Features")
    input_data = {}
    for feature in model_features:
        min_val, max_val = slider_ranges[feature]
        step = 1.0 if max_val - min_val > 1 else 0.01 if max_val - min_val > 0.1 else 0.001
        input_data[feature] = st.slider(
            label=feature.replace("_", " ").capitalize(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=step
        )

# === Prepare input data and scale ===
X_raw = pd.DataFrame([input_data])[model_features]
subset_indices = [list(scaler.feature_names_in_).index(f) for f in model_features]

subset_scaler = StandardScaler()
subset_scaler.mean_ = scaler.mean_[subset_indices]
subset_scaler.scale_ = scaler.scale_[subset_indices]
subset_scaler.feature_names_in_ = np.array(model_features)

X_scaled = subset_scaler.transform(X_raw)

# === Make prediction and generate SHAP values ===
shap_values = explainer(X_scaled)
pred_label = int(model.predict(X_scaled)[0])
proba = model.predict_proba(X_scaled)[0][1]  # Probability of class 1 ("Malignant")

# Map prediction to label
label = "Malignant 🧬" if pred_label == 1 else "Benign ✅"
color = "red" if pred_label == 1 else "green"

with right_col:
    st.markdown(f"### 🧠 Predicted Diagnosis: <span style='color:{color}; font-size: 24px'>{label}</span>", unsafe_allow_html=True)
    st.metric(label="Probability of Malignancy", value=f"{proba * 100:.2f}%")

    st.markdown("### 🔍 SHAP Waterfall Explanation")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
