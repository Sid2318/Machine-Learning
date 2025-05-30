import streamlit as st
import joblib
import numpy as np
import time as t

model = joblib.load('iris_model.pkl')

st.title("Iris Predition")

#SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
#inputs

id =  st.number_input("Enter the id",step = 1)
sepallen = st.number_input("Enter the Sepal Length(cm)",min_value=3.0, max_value=8.0,step=0.1)
sepalwid = st.number_input("Enter the Sepal Width(cm)",min_value=2.0, max_value=4.0,step=0.1)
petallen = st.number_input("Enter the Petal Length(cm)",min_value=1.0, max_value=8.0,step=0.1)
petalwid = st.number_input("Enter the Petal Width(cm)",min_value=0.0, max_value=3.0,step=0.1)

input_data = [id,sepallen,sepalwid,petallen,petalwid]
input_data = np.array(input_data)

if st.button("Predict"):
    with st.spinner():
        t.sleep(5)
    ans = model.predict(input_data.reshape(1, -1))
    if ans == 0:
        iris = "Setosa"
    elif ans == 1:
        iris = "Versicolor"
    else:
        iris = "Virginica"
    st.success(f"Iris is {iris}")
    st.balloons()

