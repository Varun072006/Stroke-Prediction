import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

df = pd.read_csv("DATASETS/final_data.csv")

X = df.drop("stroke", axis=1)
y = df["stroke"]

X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

st.set_page_config(page_title="🧠 Stroke Risk Predictor", layout="wide", page_icon="💊")

st.sidebar.title("ℹ️ About App")
st.sidebar.info(
    "This app predicts the **risk of stroke** based on healthcare data.\n"
    "Enter patient details on the left and click **Predict**.\n"
    "Built with **Streamlit + Scikit-learn**."
)

st.sidebar.markdown("---")
st.sidebar.metric("Dataset Size", f"{len(df)} Records")
st.sidebar.metric("Features Used", f"{X.shape[1]}")

st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color:#1f77b4;">🧠 Stroke Risk Prediction</h1>
        <p style="font-size:18px;">Fill the patient details below to assess the likelihood of a stroke.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    hypertension = st.radio("Hypertension", [0, 1], horizontal=True)
    heart_disease = st.radio("Heart Disease", [0, 1], horizontal=True)
    avg_glucose_level = st.slider("Average Glucose Level", 0.0, 300.0, 100.0, 0.1)
    bmi = st.slider("BMI", 0.0, 60.0, 25.0, 0.1)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Never_worked", "Private", "Self-employed", "children"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

input_data = pd.DataFrame([{
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence,
    "smoking_status": smoking_status
}])

input_data = pd.get_dummies(input_data, drop_first=True)
input_data = input_data.reindex(columns=feature_names, fill_value=0)
input_data_scaled = scaler.transform(input_data)

if st.button("🔍 Predict Stroke Risk", use_container_width=True):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Stroke Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "salmon"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke detected (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {probability:.2f})")

    st.markdown("---")
    st.write("### 🔎 Feature values used for this prediction:")
    st.dataframe(input_data, use_container_width=True)