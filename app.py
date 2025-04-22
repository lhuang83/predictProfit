import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('ProfitPredictor.pkl', 'rb'))
ohe = pickle.load(open('StateEncoder.pkl','rb'))
st.title("Profit Predictor")
rdSpend = float(st.text_input("Enter R&D Spend: ","100000"))
adminSpend = float(st.text_input("Enter Administration Spend: ","100000"))
markSpend = float(st.text_input("Enter Marketing Spend: ","100000"))
state = st.text_input("Enter State: ","New York")
stateEncoded = ohe.transform(np.array([[state]]))

features = np.concatenate((stateEncoded,np.array([[rdSpend,adminSpend,markSpend]])), axis = 1)

if st.button("Predict"):
  st.write(f"Predicted Profit is ${model.predict(features)}")
