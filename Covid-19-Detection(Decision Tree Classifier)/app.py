import streamlit as st
import joblib
import numpy as np

def encode(val):
    return 1 if val == "Yes" else 0

st.title("🦠 COVID-19 Symptom & Exposure Checker")

Breathing_Problem = st.selectbox("Breathing Problem:", ["Yes", "No"])
Fever = st.selectbox("Fever:", ["Yes", "No"])
Dry_Cough = st.selectbox("Dry Cough:", ["Yes", "No"])
Sore_throat = st.selectbox("Sore throat:", ["Yes", "No"])
Running_Nose = st.selectbox("Running Nose:", ["Yes", "No"])
Asthma = st.selectbox("Asthma:", ["Yes", "No"])
Chronic_Lung_Disease = st.selectbox("Chronic Lung Disease:", ["Yes", "No"])
Headache = st.selectbox("Headache:", ["Yes", "No"])
Heart_Disease = st.selectbox("Heart Disease:", ["Yes", "No"])
Diabetes = st.selectbox("Diabetes:", ["Yes", "No"])
Hyper_Tension = st.selectbox("Hyper Tension:", ["Yes", "No"])
Fatigue = st.selectbox("Fatigue:", ["Yes", "No"])
Gastrointestinal = st.selectbox("Gastrointestinal:", ["Yes", "No"])
Abroad_travel = st.selectbox("Abroad travel:", ["Yes", "No"])
Contact_with_COVID_Patient = st.selectbox("Contact with COVID Patient:", ["Yes", "No"])
Attended_Large_Gathering = st.selectbox("Attended Large Gathering:", ["Yes", "No"])
Visited_Public_Exposed_Places = st.selectbox("Visited Public Exposed Places:", ["Yes", "No"])
Family_working_in_Public_Exposed_Places = st.selectbox("Family working in Public Exposed Places:", ["Yes", "No"])
Wearing_Masks = st.selectbox("Wearing Masks:", ["Yes", "No"])
Sanitization_from_Market = st.selectbox("Sanitization from Market:", ["Yes", "No"])

le = joblib.load('LabelEncoder.pkl')

all_features  = [
    Breathing_Problem,
    Fever,
    Dry_Cough,
    Sore_throat,
    Running_Nose,
    Asthma,
    Chronic_Lung_Disease,
    Headache,
    Heart_Disease,
    Diabetes,
    Hyper_Tension,
    Fatigue,
    Gastrointestinal,
    Abroad_travel,
    Contact_with_COVID_Patient,
    Attended_Large_Gathering,
    Visited_Public_Exposed_Places,
    Family_working_in_Public_Exposed_Places,
    Wearing_Masks,
    Sanitization_from_Market
]

input_array = [
    encode(Breathing_Problem),
    encode(Fever),
    encode(Dry_Cough),
    encode(Sore_throat),
    encode(Running_Nose),
    encode(Headache),
    encode(Hyper_Tension),
    encode(Fatigue),
    encode(Abroad_travel),
    encode(Contact_with_COVID_Patient),
    encode(Attended_Large_Gathering),
    encode(Visited_Public_Exposed_Places),
    encode(Family_working_in_Public_Exposed_Places),
]

model = joblib.load('Covid.pkl')

if st.button("🔍 Predict COVID-19 Risk"):
    prediction = model.predict(np.array(input_array).reshape(1, -1))[0]
    
    if prediction == 1:
        st.error("⚠️ You may be at risk for COVID-19. Please consult a healthcare professional.")
    else:
        st.success("✅ You are likely safe, but continue to follow COVID-appropriate behavior.")