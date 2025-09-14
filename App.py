# app.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# -------------------------
# Config / load pipeline
# -------------------------
st.set_page_config(page_title="🧠 Depression Prediction", page_icon="🩺", layout="centered")

@st.cache_resource
def load_pipeline(path="pipe.pkl"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path.resolve()}")
    with open(path, "rb") as f:
        pipe = pickle.load(f)
    return pipe

try:
    pipe = load_pipeline("pipe.pkl")
except Exception as e:
    st.error("Could not load pipeline (pipe.pkl). Make sure file is in the same folder as this script.")
    st.exception(e)
    st.stop()

st.title("🧠 Depression Prediction")
st.write("Fill the form below. The app will preprocess your inputs exactly like the training pipeline and run the prediction.")

# -------------------------
# Form: inputs (two columns)
# -------------------------
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        profession = st.selectbox("Profession", ["Working Professional", "Student"])
        sleep = st.slider("Sleep Duration (hours)", 0, 24, 7)

        dietary = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        satisfaction = st.slider("Satisfaction (0-5)", 0, 5, 3)

    with col2:
        suicide = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
        work_hours = st.slider("Work/Study Hours (0-12)", 0, 12, 6)
        financial = st.slider("Financial Stress (0-5)", 0, 5, 2)
        family = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])
        pressure = st.slider("Pressure (0-5)", 0, 5, 2)

    st.markdown("**Optional:** adjust the decision threshold for the positive (depressed) class.")
    threshold = st.slider("Threshold (positive class)", 0.0, 1.0, 0.24, 0.01)

    submit = st.form_submit_button("🔮 Predict")

# -------------------------
# Prediction logic
# -------------------------
if submit:
    # Build DataFrame in the exact column order used during training:
    # ['Gender', 'Age', 'Working Professional or Student', 'Sleep Duration',
    #  'Dietary Habits', 'Have you ever had suicidal thoughts ?',
    #  'Work/Study Hours', 'Financial Stress',
    #  'Family History of Mental Illness', 'Pressure', 'Satisfaction']
    input_df = pd.DataFrame(
        [[gender, age, profession, sleep, dietary, suicide, work_hours, financial, family, pressure, satisfaction]],
        columns=[
            'Gender', 'Age', 'Working Professional or Student', 'Sleep Duration',
            'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Work/Study Hours', 'Financial Stress',
            'Family History of Mental Illness', 'Pressure', 'Satisfaction'
        ],
    )

    st.markdown("### Input preview")
    st.dataframe(input_df, use_container_width=True)

    # Run prediction & probability (if available)
    try:
        # Prefer predict_proba if pipeline supports it (most pipelines do if final estimator does)
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(input_df)
        else:
            # Some pipelines encapsulate the model in named_steps; but pipeline.predict_proba should exist.
            # Fallback: transform with preprocessing steps then call model.predict_proba if available
            try:
                # attempt to access the last step (estimator) and intermediate preprocessing
                last_step = None
                if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
                    last_step = pipe.named_steps["model"]
                elif isinstance(pipe, (list, tuple)):
                    last_step = pipe[-1][1]
                # fallback if transform available
                if last_step is not None and hasattr(last_step, "predict_proba"):
                    # transform using all steps except last
                    preproc = pipe[:-1]
                    X_trans = preproc.transform(input_df)
                    proba = last_step.predict_proba(X_trans)
            except Exception:
                proba = None

        if proba is not None:
            # binary case: assume positive class index is 1
            if proba.shape[1] == 2:
                pos_prob = float(proba[0, 1])
            else:
                # if single column returned, assume it is prob of positive
                pos_prob = float(proba[0, 0])

            pred_label = 1 if pos_prob >= threshold else 0
            label_str = "Depressed" if pred_label == 1 else "Not Depressed"

            # Show result
            st.markdown("### Prediction")
            if pred_label == 1:
                st.error(f"⚠️ {label_str}")
            else:
                st.success(f"✅ {label_str}")

            st.markdown(f"**Confidence (positive class):** {pos_prob:.2%}")

            # Nice visual: progress bar showing confidence
            st.progress(min(max(pos_prob, 0.0), 1.0))

        else:
            # If no probability available, use predict()
            pred = pipe.predict(input_df)[0]
            label_str = "Depressed" if int(pred) == 1 else "Not Depressed"
            st.markdown("### Prediction")
            if int(pred) == 1:
                st.error(f"⚠️ {label_str}")
            else:
                st.success(f"✅ {label_str}")

        # Optional: show raw model outputs for debugging
        if st.checkbox("Show raw model output"):
            try:
                st.write("predict:", pipe.predict(input_df))
                if hasattr(pipe, "predict_proba"):
                    st.write("predict_proba:", pipe.predict_proba(input_df))
            except Exception as ex:
                st.write("Could not show raw outputs:", ex)

    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)

st.write("---")

