import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained model
model = load("adult_income_model.joblib")

st.title("Adult Income Prediction")
st.write("Predict whether income is >50K or <=50K")

# ---- User Inputs ----
age = st.number_input("Age", min_value=17, max_value=90, value=30)
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
education_num = st.number_input("Education Number", min_value=1, max_value=16, value=9)

sex = st.selectbox("Sex", ["Male", "Female"])

workclass = st.selectbox(
    "Workclass",
    ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
     "Local-gov", "State-gov", "Without-pay", "Never-worked"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married-civ-spouse", "Divorced", "Never-married",
     "Separated", "Widowed", "Married-spouse-absent"]
)

occupation = st.selectbox(
    "Occupation",
    ["Tech-support", "Craft-repair", "Other-service", "Sales",
     "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
     "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
     "Transport-moving", "Priv-house-serv", "Protective-serv",
     "Armed-Forces"]
)

relationship = st.selectbox(
    "Relationship",
    ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
)

# ---- Build Input DataFrame ----
input_df = pd.DataFrame({
    "age": [age],
    "education-num": [education_num],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss],
    "hours-per-week": [hours_per_week],
    "sex": [1 if sex == "Male" else 0],
    "workclass": [workclass],
    "marital.status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship]
})

# ---- One-Hot Encoding (same as training) ----
input_df = pd.get_dummies(
    input_df,
    columns=['workclass','marital.status','occupation','relationship'],
    drop_first=True,
    dtype=int
)

# ---- Align columns with training model ----
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# ---- Prediction ----
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(
        f"The model predicts that the person's income is greater than 50,000 USD "
        f"with a confidence of {prob:.2f}."
    )
    else:
        st.info(
        f"The model predicts that the person's income is less than or equal to 50,000 USD "
        f"with a confidence of {prob:.2f}."
    )
