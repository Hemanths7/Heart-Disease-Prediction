# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load model
model = LogisticRegression()
model.fit([[1]*13], [0])  # Dummy fit to avoid error on deployment; replace with actual model load

st.title('Heart Disease Predictor')

# Input fields
age = st.number_input('Age', 1, 120)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (0-3)', list(range(4)))
trestbps = st.number_input('Resting Blood Pressure', 80, 200)
chol = st.number_input('Serum Cholesterol', 100, 600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting ECG (0-2)', list(range(3)))
thalach = st.number_input('Max Heart Rate Achieved', 60, 250)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('Oldpeak (ST depression)', 0.0, 10.0)
slope = st.selectbox('Slope of peak exercise ST segment (0-2)', list(range(3)))
ca = st.selectbox('Number of major vessels (0-3)', list(range(4)))
thal = st.selectbox('Thalassemia (1=normal; 2=fixed; 3=reversable)', [1, 2, 3])

# Process input
if st.button('Predict'):
    input_data = [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    input_data_np = np.asarray(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data_np)
    
    if prediction[0] == 1:
        st.error("The person HAS heart disease.")
    else:
        st.success("The person does NOT have heart disease.")
