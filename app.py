"""
app.py
------
Streamlit web app for Heart Failure Prediction.
Loads the pre-trained XGBoost model from ./models/ and lets users
input clinical features to receive a heart disease risk estimate.

Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)


# ------------------------------------------------------------------
# Load model (cached so it loads only once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    return scaler, feature_names, xgb_model


# ------------------------------------------------------------------
# Feature engineering — mirrors train.py exactly
# ------------------------------------------------------------------
def build_feature_row(inputs, feature_names):
    """
    Takes the raw user inputs dict, applies the same preprocessing
    and feature engineering used during training, and returns a
    DataFrame with one row aligned to feature_names.
    """
    row = {}

    # Numeric
    row["Age"]         = inputs["Age"]
    row["RestingBP"]   = inputs["RestingBP"]
    row["Cholesterol"] = inputs["Cholesterol"]
    row["FastingBS"]   = int(inputs["FastingBS"])
    row["MaxHR"]       = inputs["MaxHR"]
    row["Oldpeak"]     = inputs["Oldpeak"]

    # Binary encoded
    row["Sex"]            = 1 if inputs["Sex"] == "Male" else 0
    row["ExerciseAngina"] = 1 if inputs["ExerciseAngina"] == "Yes" else 0

    # One-hot: ChestPainType
    for val in ["ASY", "ATA", "NAP", "TA"]:
        row[f"ChestPainType_{val}"] = 1 if inputs["ChestPainType"] == val else 0

    # One-hot: RestingECG
    for val in ["LVH", "Normal", "ST"]:
        row[f"RestingECG_{val}"] = 1 if inputs["RestingECG"] == val else 0

    # One-hot: ST_Slope
    for val in ["Down", "Flat", "Up"]:
        row[f"ST_Slope_{val}"] = 1 if inputs["ST_Slope"] == val else 0

    # Engineered features
    row["HR_Efficiency"] = inputs["MaxHR"] / (220 - inputs["Age"])

    slope_weight = 0
    if inputs["ST_Slope"] == "Flat":
        slope_weight = 1
    elif inputs["ST_Slope"] == "Down":
        slope_weight = 2
    row["EKG_Score"]     = inputs["Oldpeak"] + slope_weight
    row["Stress_Impact"] = inputs["Oldpeak"] + 2 * row["ExerciseAngina"]

    # Align to training column order
    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_names, fill_value=0)
    return df_row


# ------------------------------------------------------------------
# App UI
# ------------------------------------------------------------------
def main():
    st.title("❤️ Heart Failure Prediction")
    st.markdown(
        "Enter a patient's clinical information below to receive a "
        "heart disease risk estimate from the model."
    )
    st.markdown("---")

    # Load model
    try:
        scaler, feature_names, xgb_model = load_model()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `python train.py` first "
            "to train and save the models, then restart this app."
        )
        st.stop()

    # ------------------------------------------------------------------
    # Instructions expander
    # ------------------------------------------------------------------
    with st.expander("How to use this app — click to expand"):
        st.markdown("""
**Step 1 — Fill in the patient information** using the fields below.
All fields are required. Hover over any field label to see a description of what it means.

**Step 2 — Click "Predict"** at the bottom of the form.

**Step 3 — Read the result.** The model outputs a risk probability (0% to 100%), interpreted as follows:

- **Below 30%** — Low Risk: The model sees little evidence of heart disease.
- **30% to 60%** — Borderline: The result is uncertain. Clinical follow-up is advisable.
- **Above 60%** — High Risk: The model finds significant indicators of heart disease.

A result in the borderline range does not mean the patient is safe — it means the model is not confident either way, and medical evaluation is warranted.

---

**Medical Disclaimer:** This tool is a student research project built for educational purposes only.
It is not a medical device and should not be used for clinical diagnosis or treatment decisions.
Always consult a licensed healthcare professional.
        """)

    st.markdown("### Patient Information")

    # ------------------------------------------------------------------
    # Input form
    # ------------------------------------------------------------------
    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input(
                "Age (years)",
                min_value=20, max_value=100, value=54,
                help="Patient age in years. Dataset range: 28–77."
            )
            sex = st.selectbox(
                "Sex",
                options=["Male", "Female"],
                help="Biological sex of the patient."
            )
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=["No", "Yes"],
                help="Whether fasting blood sugar exceeds 120 mg/dL. Elevated levels may indicate diabetes or pre-diabetes."
            )

            st.markdown("**Vitals**")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=50, max_value=250, value=130,
                help="Blood pressure measured at rest. Normal adult range: 90–120 mmHg systolic."
            )
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=0, max_value=700, value=240,
                help="Total serum cholesterol. Normal range: 125–200 mg/dL. Enter 0 if unknown — it will be estimated from typical values."
            )
            max_hr = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=50, max_value=220, value=140,
                help="Highest heart rate recorded during exercise testing. Declines naturally with age."
            )

        with col2:
            st.markdown("**Clinical Findings**")
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=["ASY", "ATA", "NAP", "TA"],
                help=(
                    "ASY = Asymptomatic (no chest pain — paradoxically the highest-risk type)  \n"
                    "ATA = Atypical Angina  \n"
                    "NAP = Non-Anginal Pain  \n"
                    "TA  = Typical Angina"
                )
            )
            resting_ecg = st.selectbox(
                "Resting ECG Result",
                options=["Normal", "ST", "LVH"],
                help=(
                    "Normal = Normal ECG  \n"
                    "ST = ST-T wave abnormality (possible ischemia)  \n"
                    "LVH = Left Ventricular Hypertrophy (enlarged heart muscle)"
                )
            )
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                options=["No", "Yes"],
                help="Whether the patient experiences chest pain or pressure during exercise. A strong risk indicator."
            )
            oldpeak = st.number_input(
                "Oldpeak (ST Depression)",
                min_value=-3.0, max_value=7.0, value=0.0, step=0.1,
                help="ST depression induced by exercise relative to rest. Higher values indicate more severe cardiac stress."
            )
            st_slope = st.selectbox(
                "ST Slope",
                options=["Up", "Flat", "Down"],
                help=(
                    "Slope of the peak exercise ST segment.  \n"
                    "Up = Upsloping (generally healthy)  \n"
                    "Flat = Flat (moderate risk)  \n"
                    "Down = Downsloping (highest risk)"
                )
            )

        st.markdown("---")
        submitted = st.form_submit_button("Predict", use_container_width=True)

    # ------------------------------------------------------------------
    # Prediction and Results
    # ------------------------------------------------------------------
    if submitted:
        # Handle cholesterol = 0 (impute with training medians by sex)
        chol_value = cholesterol if cholesterol > 0 else (239.0 if sex == "Male" else 238.0)

        inputs = {
            "Age":            age,
            "Sex":            sex,
            "FastingBS":      1 if fasting_bs == "Yes" else 0,
            "RestingBP":      resting_bp,
            "Cholesterol":    chol_value,
            "MaxHR":          max_hr,
            "ChestPainType":  chest_pain,
            "RestingECG":     resting_ecg,
            "ExerciseAngina": exercise_angina,
            "Oldpeak":        oldpeak,
            "ST_Slope":       st_slope,
        }

        df_row = build_feature_row(inputs, feature_names)
        prob   = float(xgb_model.predict_proba(df_row.values)[0][1])
        pct    = prob * 100

        st.markdown("---")
        st.markdown("## Result")

        # Determine risk tier
        if prob >= 0.60:
            risk_label  = "High Risk"
            label_color = "red"
        elif prob >= 0.30:
            risk_label  = "Borderline"
            label_color = "orange"
        else:
            risk_label  = "Low Risk"
            label_color = "green"

        st.markdown(
            f"<h3 style='color:{label_color}'>{risk_label} — {pct:.1f}%</h3>",
            unsafe_allow_html=True
        )

        # Progress bar — must be a plain Python float in [0.0, 1.0]
        st.progress(float(prob))

        st.markdown("#### Risk Interpretation")
        if prob >= 0.60:
            st.error(
                "The model finds significant indicators of heart disease. "
                "Prompt further cardiac evaluation is strongly recommended."
            )
        elif prob >= 0.30:
            st.warning(
                "The result falls in the borderline range — the model is not confident either way. "
                "This does not mean the patient is safe. Clinical follow-up and further testing are advisable."
            )
        else:
            st.success(
                "The model sees little evidence of heart disease based on the provided features. "
                "Routine monitoring remains a good practice."
            )

        st.markdown("---")
        st.caption(
            "This tool is a student research project for educational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment."
        )


if __name__ == "__main__":
    main()
