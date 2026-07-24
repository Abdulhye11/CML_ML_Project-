
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CML Screening",
    page_icon="🩸",
    layout="wide"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_resources():
    model = joblib.load("cml_random_forest_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    label_mapping = joblib.load("label_mapping.pkl")
    return model, feature_names, label_mapping

model, feature_names, label_mapping = load_resources()

# ==========================================
# HEADER
# ==========================================

st.title("🩸 Early Screening of Chronic Myeloid Leukemia")

st.markdown("""
This application predicts possible blood disorders using **Complete Blood Count (CBC)** parameters and a trained **Random Forest** model.

⚠️ **Disclaimer:** This application is for educational and research purposes only.
It must **not** be used as a substitute for professional medical advice.
""")

st.divider()
