# modules/page_prediction.py
"""
Page 4: Prediction & What-If Analysis
Interactive risk prediction using the trained neural network and local sensitivity scenarios.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from inference.import_model import HeartDiseaseNNModel


@st.cache_resource(show_spinner="Loading trained neural network model...")
def load_model() -> HeartDiseaseNNModel:
    """Load the pretrained heart disease model as a cached resource."""
    return HeartDiseaseNNModel.from_pretrained()


def build_input_vector(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
) -> np.ndarray:
    """
    Build an input vector in the exact feature order expected by the model.
    """
    return np.array(
        [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ],
        dtype=float,
    ).reshape(1, -1)


def show():
    """Render the Prediction & What-If Analysis page."""
    st.header("🤖 Heart Disease Risk Prediction & What-If Analysis")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No data loaded. Please go to the Dataset Overview page first.")
        return

    df = st.session_state["df"]
    model = load_model()

    st.markdown(
        """
This page allows you to **simulate patient profiles**, obtain a predicted
probability of heart disease, and run simple *what-if* scenarios to understand
how changes in clinical variables may affect the risk.
        """
    )

    # --- PATIENT INPUT FORM ---
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 29, 77, 54)
            sex_label = st.selectbox("Sex", options=["Male", "Female"], index=0)
            sex = 1 if sex_label == "Male" else 0

            cp = st.selectbox(
                "Chest pain type (cp)",
                options=[0, 1, 2, 3],
                format_func=lambda v: {
                    0: "0 - Typical angina",
                    1: "1 - Atypical angina",
                    2: "2 - Non-anginal pain",
                    3: "3 - Asymptomatic",
                }[v],
            )

            exang_label = st.selectbox(
                "Exercise induced angina (exang)", options=["No", "Yes"], index=0
            )
            exang = 1 if exang_label == "Yes" else 0

        with col2:
            trestbps = st.slider("Resting blood pressure (trestbps)", 90, 200, 130)
            chol = st.slider("Cholesterol (chol)", 120, 564, 240)
            thalach = st.slider("Max heart rate (thalach)", 70, 200, 150)
            fbs_label = st.selectbox(
                "Fasting blood sugar > 120 mg/dl (fbs)", options=["No", "Yes"], index=0
            )
            fbs = 1 if fbs_label == "Yes" else 0

        with col3:
            restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
            oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox("Slope of ST segment (slope)", options=[0, 1, 2])
            ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])

        submitted = st.form_submit_button("🔮 Predict Risk")

    if not submitted:
        st.info("Configure the patient profile and click **Predict Risk**.")
        return

    # --- BASE PREDICTION ---
    x_base = build_input_vector(
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    )
    proba = model.predict_proba(x_base)[0, 1]  # probability of class 1 (disease)
    pred_class = model.predict(x_base)[0]

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("📌 Predicted Risk for Current Profile")
        st.metric(
            label="Estimated probability of heart disease",
            value=f"{proba*100:.1f}%",
            delta=None,
        )

        st.markdown(
            f"""
**Model decision:** `{"High risk (class 1)" if pred_class == 1 else "Low risk (class 0)"}`  
Current profile is interpreted as a patient with an estimated risk of **{proba*100:.1f}%**.
            """
        )

    with col_side:
        st.markdown("### ℹ️ Legend")
        st.write("- **Class 1** → Heart disease present")
        st.write("- **Class 0** → No heart disease")

    # --- WHAT-IF ANALYSIS ---
    st.markdown("---")
    st.subheader("🧪 What-If Analysis (Local Sensitivity)")

    st.markdown(
        """
We keep the current patient profile fixed and modify one variable at a time
to see how the predicted probability changes.
        """
    )

    scenarios = []

    # 1) Age - 10 years (if possible)
    if age - 10 >= 29:
        x_young = build_input_vector(
            age - 10, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal
        )
        p_young = model.predict_proba(x_young)[0, 1]
        scenarios.append(("Age - 10 years", p_young))

    # 2) Normalize cholesterol (set to 200)
    x_chol_200 = build_input_vector(
        age, sex, cp, trestbps, 200, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    )
    p_chol_200 = model.predict_proba(x_chol_200)[0, 1]
    scenarios.append(("Cholesterol = 200 mg/dl", p_chol_200))

    # 3) No exercise induced angina
    x_no_exang = build_input_vector(
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, 0, oldpeak, slope, ca, thal
    )
    p_no_exang = model.predict_proba(x_no_exang)[0, 1]
    scenarios.append(("No exercise-induced angina (exang=0)", p_no_exang))

    # 4) Lower ST depression
    x_low_oldpeak = build_input_vector(
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, max(0.0, oldpeak - 1.0), slope, ca, thal
    )
    p_low_oldpeak = model.predict_proba(x_low_oldpeak)[0, 1]
    scenarios.append(("Lower ST depression (oldpeak - 1.0)", p_low_oldpeak))

    # Build comparison table
    rows = [("Current profile", proba)] + scenarios
    df_whatif = pd.DataFrame(
        rows, columns=["Scenario", "Probability of disease"]
    )
    df_whatif["Probability of disease"] = df_whatif["Probability of disease"].apply(
        lambda p: f"{p*100:.1f}%"
    )

    st.table(df_whatif)

    st.markdown(
        """
This simple local sensitivity analysis helps stakeholders understand which clinical
variables have the largest impact on the **individual risk** for this specific patient.
        """
    )