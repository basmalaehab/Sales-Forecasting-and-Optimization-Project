import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load saved model and encoders
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("Sales Forecasting with XGBoost")
st.write("This app predicts **sales** based on the input features.")

# Select input options from encoder classes
ship_mode = st.selectbox("Ship Mode", label_encoders['Ship Mode'].classes_)
segment = st.selectbox("Segment", label_encoders['Segment'].classes_)
order_priority = st.selectbox("Order Priority", label_encoders['Order Priority'].classes_)
city = st.selectbox("City", label_encoders['City'].classes_)
sub_category = st.selectbox("Sub-Category", label_encoders['Sub-Category'].classes_)

order_date = st.date_input("Order Date", value=datetime(2016, 1, 1))
ship_date = st.date_input("Ship Date", value=datetime(2016, 1, 3))

if ship_date <= order_date:
    st.error("🚫 Ship Date must be **after** Order Date.")
    st.stop()

order_date_ts = int(pd.to_datetime(order_date).timestamp())
ship_date_ts = int(pd.to_datetime(ship_date).timestamp())

profit = st.number_input("Profit")
quantity = st.number_input("Quantity", min_value=1, step=1, value=5)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
shipping_cost = st.number_input("Shipping Cost", min_value=0.0, step=0.1, value=10.0)

def encode(col, val):
    try:
        return label_encoders[col].transform([val])[0]
    except ValueError:
        st.error(f"Value '{val}' not found in encoder for '{col}'")
        st.stop()

input_vector = [
    order_date_ts,
    ship_date_ts,
    encode('Ship Mode', ship_mode),
    encode('Segment', segment),
    encode('City', city),
    encode('Sub-Category', sub_category),
    profit,
    quantity,
    discount,
    shipping_cost,
    encode('Order Priority', order_priority),
]

input_features = np.array([input_vector])

if st.button("Predict"):
    try:
        prediction = model.predict(input_features)[0]
        st.success(f"The predicted sales is: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
