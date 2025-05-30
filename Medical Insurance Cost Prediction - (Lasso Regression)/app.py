import streamlit as st
import joblib
import numpy as np
import time as t

#load model
model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl') 
#You need to make sure that the input to the model has the same structure and number of features as the training data used during model fitting.

st.title('🩺 Medical Insurance Cost Prediction')

# age,sex,bmi,children,smoker,region,charges

#input
age = st.number_input("Enter ur age:",min_value= 18,max_value= 120,step = 1)
sex = st.selectbox("Enter ur sex:",["Male","Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10,step =1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert inputs to model-ready format
input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "sex_female": 1 if sex == "female" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "smoker_no": 1 if smoker == "no" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
    "region_northeast": 1 if region == "northeast" else 0,
}

# Convert to array and scale
input_df = np.array([[
    input_data["age"],
    input_data["bmi"],
    input_data["children"],
    input_data["sex_male"],
    input_data["smoker_yes"],
    input_data["sex_female"],
    input_data["smoker_no"],
    input_data["region_northwest"],
    input_data["region_southeast"],
    input_data["region_southwest"],
    input_data["region_northeast"],
]])
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Charges 💸"):
    prediction = model.predict(scaled_input)
    with st.spinner("Just wait"):
        t.sleep(5)
        st.balloons()
        st.success(f"Estimated Insurance Charges: ₹{prediction[0]:,.2f}")
