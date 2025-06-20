import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = joblib.load('model.pkl')
le = joblib.load('le.pkl')

st.title("🎾 Play Tennis Predictor")

outlook = st.selectbox("Outlook ☁️:", options=['Sunny', 'Overcast', 'Rain'])
temp = st.selectbox("Temperature 🌡️:", options=['Hot', 'Mild', 'Cool'])
humidity = st.selectbox("Humidity 💧:", options=['High', 'Normal'])
wind = st.selectbox("Wind 🌬️:", options=['Weak', 'Strong'])

input = [[outlook, temp, humidity, wind]]
le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()

le_outlook.fit(['Sunny', 'Overcast', 'Rain'])
le_temp.fit(['Hot', 'Mild', 'Cool'])
le_humidity.fit(['High', 'Normal'])
le_wind.fit(['Weak', 'Strong'])

le = [le_outlook, le_temp, le_humidity, le_wind]
encoded = [enc.transform([val])[0] for enc, val in zip(le, input[0])]

# Convert to 2D array for model.predict
encoded = np.array(encoded).reshape(1, -1)

if(st.button("Predict play")):
    ans = model.predict(encoded)[0]
    if ans == 1:
        st.success("Yes")
        st.balloons()
    else:
        st.error("No")