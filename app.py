import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('ProfitPredictor.pkl', 'rb'))
ohe = pickle.load(open('StateEncoder.pkl','rb'))
st.title("Profit Predictor")
rdSpend = np.array([[float(st.text_input("Enter R&D Spend: ","100000"))]])
adminSpend = np.array([[float(st.text_input("Enter Administration Spend: ","100000"))]])
markSpend = np.array([[float(st.text_input("Enter Marketing Spend: ","100000"))]])
state = np.array([[st.text_input("Enter State: ","New York")]])
state = ohe.transform([[state]])

features = np.concatenate((state,np.array([[rdSpend,admSpend,markSpend]])),axis = 1)

if st.button("Predict"):
  st.write(f"Predicted Profit is ${model.predict(features)}")