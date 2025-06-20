import streamlit as st
import joblib

model = joblib.load('model.pkl')
st.title("Hello Glass Type Prediction! 👋")

Ri = st.number_input("Refractive Index (RI)", min_value=1.5111, max_value=1.5339, value=1.5184, step=0.0001)
Na = st.number_input("Sodium (Na)", min_value=10.73, max_value=17.38, value=13.41, step=0.01)
Mg = st.number_input("Magnesium (Mg)", min_value=0.0, max_value=4.49, value=2.68, step=0.01)
Al = st.number_input("Aluminum (Al)", min_value=0.29, max_value=3.5, value=1.44, step=0.01)
Si = st.number_input("Silicon (Si)", min_value=69.81, max_value=75.41, value=72.65, step=0.01)
k  = st.number_input("Potassium (K)", min_value=0.0, max_value=6.21, value=0.50, step=0.01)
Ca = st.number_input("Calcium (Ca)", min_value=5.43, max_value=16.19, value=8.96, step=0.01)
Ba = st.number_input("Barium (Ba)", min_value=0.0, max_value=3.15, value=0.18, step=0.01)
Fe = st.number_input("Iron (Fe)", min_value=0.0, max_value=0.51, value=0.057, step=0.001)

if st.button("Submit"):
    input_data = [[Ri, Na, Mg, Al, Si, k, Ca, Ba, Fe]]
    st.success(f"Type {model.predict(input_data)}")
    st.balloons()