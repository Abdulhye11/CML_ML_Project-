import plotly.express as px
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
st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F7FAFC;
}

/* Buttons */
.stButton > button{
    width:100%;
    background:#1565C0;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

/* Input boxes */
.stNumberInput,
.stSelectbox{
    background:white;
}

/* Headers */
h1{
    color:#1565C0;
}

h2{
    color:#0D47A1;
}

</style>
""", unsafe_allow_html=True)

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

st.title("🩸 AI-Based CML Screening System")

st.caption(
    "Machine Learning Based Early Detection using Complete Blood Count (CBC)"
)

st.markdown("""
This application predicts possible blood disorders using **Complete Blood Count (CBC)** parameters and a trained **Random Forest** model.

⚠️ **Disclaimer:** This application is for educational and research purposes only.
It must **not** be used as a substitute for professional medical advice.
""")

st.divider()
# ==========================================
# PATIENT INFORMATION
# ==========================================

st.sidebar.header("👤 Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=40
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender_value = 1 if gender == "Male" else 0

severity = st.sidebar.number_input(
    "Severity Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
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
    # ==========================================
# PREDICTION
# ==========================================

st.divider()

if st.button("🔍 Predict Disease"):

    input_data = pd.DataFrame(
        [[
            age,
            gender_value,
            severity,
            WBC,
            RBC,
            Hemoglobin,
            Hematocrit,
            Platelets,
            MCV,
            MCH,
            MCHC,
            RDW,
            Neutrophils,
            Lymphocytes,
            Monocytes,
            Eosinophils,
            Basophils
        ]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = probability.max() * 100

    disease = label_mapping[prediction]

    col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🩸 Prediction",
        disease
    )

with col2:
    st.metric(
        "🎯 Confidence",
        f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )

    if confidence < 70:
        st.warning(
            "Low confidence prediction. Please interpret results carefully and consult a healthcare professional."
        )
    else:
        st.success(
            "The model shows a strong pattern match."
        )

    # ==========================================
    # PROBABILITY DISPLAY
    # ==========================================

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame(
        {
            "Disease": list(label_mapping.values()),
            "Probability (%)": probability * 100
        }
    )

    probability_df = probability_df.sort_values(
        by="Probability (%)",
        ascending=False
    )

    st.dataframe(
        probability_df.style.format(
            {"Probability (%)": "{:.2f}"}
        ),
        use_container_width=True
    )
    st.subheader("📊 Prediction Probability Chart")

fig = px.bar(
    probability_df,
    x="Disease",
    y="Probability (%)",
    text="Probability (%)",
    color_discrete_sequence=["#1565C0"]
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    title="Prediction Probability",
    xaxis_title="Disease",
    yaxis_title="Probability (%)",
    showlegend=False,
    height=450
)

st.plotly_chart(fig, use_container_width=True)
fig.update_layout(
    xaxis_title="Disease",
    yaxis_title="Probability (%)",
    showlegend=False,
    height=450
)

st.plotly_chart(fig, use_container_width=True)
