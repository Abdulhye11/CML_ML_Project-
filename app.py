
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
# ==========================================
# PATIENT INFORMATION
# ==========================================

st.header("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

gender_value = 1 if gender == "Male" else 0
# ==========================================
# CBC PARAMETERS
# ==========================================

st.header("🩸 Complete Blood Count (CBC)")

col1, col2, col3 = st.columns(3)

with col1:
    WBC = st.number_input("WBC", value=7.0)
    RBC = st.number_input("RBC", value=4.8)
    Hemoglobin = st.number_input("Hemoglobin", value=14.5)
    Hematocrit = st.number_input("Hematocrit", value=42.0)
    Platelets = st.number_input("Platelets", value=250)

with col2:
    MCV = st.number_input("MCV", value=90.0)
    MCH = st.number_input("MCH", value=30.0)
    MCHC = st.number_input("MCHC", value=33.0)
    RDW = st.number_input("RDW", value=13.0)

with col3:
    Neutrophils = st.number_input("Neutrophils", value=60.0)
    Lymphocytes = st.number_input("Lymphocytes", value=30.0)
    Monocytes = st.number_input("Monocytes", value=5.0)
    Eosinophils = st.number_input("Eosinophils", value=3.0)
    Basophils = st.number_input("Basophils", value=1.0)
