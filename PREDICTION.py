import mlflow  # type: ignore
import mlflow.xgboost  # type: ignore
import pandas as pd  # type: ignore
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

if ship_date <= order_date:
    st.error("🚫 Ship Date must be **after** Order Date.")
    st.stop()

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

# Feature vector
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

# Predict button
if st.button("Predict"):
    try:
        # Make prediction
        prediction = model.predict(input_features)[0]
        st.success("Prediction complete!")
        st.write(f"The predicted **sales** is: **${prediction:.2f}**")

        # Ask for actual sales
        actual_sales = st.number_input("Enter the actual sales value (after you get it):", min_value=0.0)

        # Define performance threshold
        threshold = 1000  # Customize this as needed

        # Start MLflow run
        with mlflow.start_run(run_name="user_prediction"):
            # Log input parameters
            mlflow.log_param("order_date", str(order_date))
            mlflow.log_param("ship_date", str(ship_date))
            mlflow.log_param("ship_mode", ship_mode)
            mlflow.log_param("segment", segment)
            mlflow.log_param("order_priority", order_priority)
            mlflow.log_param("city", city)
            mlflow.log_param("sub_category", sub_category)
            mlflow.log_param("profit", profit)
            mlflow.log_param("quantity", quantity)
            mlflow.log_param("discount", discount)
            mlflow.log_param("shipping_cost", shipping_cost)

            # Log prediction
            mlflow.log_metric("predicted_sales", prediction)

            # Save input as artifact
            input_df = pd.DataFrame([input_vector], columns=[
                "order_date_ts", "ship_date_ts", "ship_mode", "segment", "city",
                "sub_category", "profit", "quantity", "discount",
                "shipping_cost", "order_priority"
            ])
            input_df.to_csv("user_input.csv", index=False)
            mlflow.log_artifact("user_input.csv")

            # Optional: Feedback loop if actual sales entered
            if actual_sales > 0:
                error = abs(actual_sales - prediction)
                st.write(f"Absolute Error: **${error:.2f}**")
                mlflow.log_param("actual_sales", actual_sales)
                mlflow.log_metric("absolute_error", error)

                # Alert system
                if error > threshold:
                    st.error(f"ALERT: Accuracy dropped! Error = ${error:.2f} exceeds threshold (${threshold}).")
                    # You can add email/Slack alert integrations here if needed.
                    mlflow.log_param("alert_triggered", True)
                else:
                    mlflow.log_param("alert_triggered", False)

                st.success("Feedback logged in MLflow!")

        st.write("Thank you for using the Sales Forecasting app!")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
