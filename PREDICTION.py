import mlflow  # type:ignore
import mlflow.xgboost  # type:ignore
import pandas as pd  # type:ignore
import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# Initialize MLflow tracking
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sales_Forecasting_and_Optimization")

# Load model
model_version = 10
model_name = "XGBoost_Sales_Forecasting_Model"
model = mlflow.xgboost.load_model(f"models:/{model_name}/{model_version}")



# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Streamlit UI
st.title("Sales Forecasting with XGBoost")
st.write("This app predicts **sales** based on the input features.")

# Categorical inputs
ship_mode = st.selectbox("Ship Mode", label_encoders['Ship Mode'].classes_)
segment = st.selectbox("Segment", label_encoders['Segment'].classes_)
order_priority = st.selectbox("Order Priority", label_encoders['Order Priority'].classes_)
city = st.selectbox("City", label_encoders['City'].classes_)
sub_category = st.selectbox("Sub-Category", label_encoders['Sub-Category'].classes_)


# Date inputs
order_date = st.date_input("Order Date", value=datetime(2016, 1, 1))
ship_date = st.date_input("Ship Date", value=datetime(2016, 1, 3))

# Validate dates
if ship_date <= order_date:
    st.error("🚫 Ship Date must be **after** Order Date.")
    st.stop()

# Convert dates to timestamps
order_date_ts = int(pd.to_datetime(order_date).timestamp())
ship_date_ts = int(pd.to_datetime(ship_date).timestamp())


# Numerical inputs
profit = st.number_input("Profit")
quantity = st.number_input("Quantity", min_value=1, step=1, value=5)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
shipping_cost = st.number_input("Shipping Cost", min_value=0.0, step=0.1, value=10.0)


# Encoding function
def encode(col, val):
    try:
        return label_encoders[col].transform([val])[0]
    except ValueError:
        st.error(f"Value '{val}' not found in encoder for '{col}'")
        st.stop()

# Build input feature vector
input_features = np.array([[
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
]])

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_features)
        st.success(" Prediction complete!")
        st.write(f" The predicted **sales** is: **${prediction[0]:.2f}**")
        st.write(" Thank you for using the Sales Forecasting app!")
    except Exception as e:
        st.error(f" Prediction failed: {str(e)}")