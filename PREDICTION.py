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
model_version = 7
model_name = "XGBoost_Sales_Forecasting_Model"
model = mlflow.xgboost.load_model(f"models:/{model_name}/{model_version}")

# Load data
df = pd.read_csv("global_superstore_2016_cleaned_data.csv")

# Create mappings
city_mapping_df = df[['City', 'State', 'Country', 'Region', 'Market']].drop_duplicates(subset=['City'])
sub_category_mapping_df = df[['Sub-Category', 'Category']].drop_duplicates(subset=['Sub-Category'])

# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Streamlit UI
st.title("Sales Forecasting with XGBoost")
st.write("This app predicts **sales** based on the input features.")

# Categorical inputs
ship_mode = st.selectbox("Ship Mode", label_encoders['Ship Mode'].classes_)
segment = st.selectbox("Segment", label_encoders['Segment'].classes_)

# City and auto-fill fields
city = st.selectbox("City", city_mapping_df['City'].unique())
selected_row = city_mapping_df[city_mapping_df['City'] == city].iloc[0]
state = selected_row['State']
country = selected_row['Country']
region = selected_row['Region']
market = selected_row['Market']
st.selectbox("State", [state], disabled=True)
st.selectbox("Country", [country], disabled=True)
st.selectbox("Region", [region], disabled=True)
st.selectbox("Market", [market], disabled=True)

# Sub-Category and auto-fill Category
sub_category = st.selectbox("Sub-Category", sub_category_mapping_df['Sub-Category'].unique())
category = sub_category_mapping_df[sub_category_mapping_df['Sub-Category'] == sub_category].iloc[0]['Category']
st.selectbox("Category", [category], disabled=True)

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

# Determine order season
def get_order_season(order_date):
    month = order_date.month
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Fall"

order_season = get_order_season(order_date)
st.selectbox("Order Season", [order_season], disabled=True)

# Other categorical inputs
order_priority = st.selectbox("Order Priority", label_encoders['Order Priority'].classes_)

# Numerical inputs
profit = st.number_input("Profit")
quantity = st.number_input("Quantity", min_value=1, step=1, value=5)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
shipping_cost = st.number_input("Shipping Cost", min_value=0.0, step=0.1, value=10.0)

# Date-based features
order_date_obj = pd.to_datetime(order_date)
day_of_week = order_date_obj.strftime('%A')

is_black_friday = 1 if (order_date_obj.month == 11 and order_date_obj.day >= 23) else 0
is_black_friday_display = 'Yes' if is_black_friday else 'No'

is_weekend = 1 if (order_date_obj.weekday() >= 5) else 0

st.selectbox("Day of Week", [day_of_week], disabled=True)
st.selectbox("Is Black Friday?", ['Yes', 'No'], index=['Yes', 'No'].index(is_black_friday_display), disabled=True)
st.selectbox("Is Weekend?", ['No', 'Yes'], index=is_weekend, disabled=True)

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
    encode('State', state),
    encode('Country', country),
    encode('Region', region),
    encode('Market', market),
    encode('Category', category),
    encode('Sub-Category', sub_category),
    profit,
    quantity,
    discount,
    shipping_cost,
    encode('Order Priority', order_priority),
    encode('Order_Season', order_season),
    is_black_friday,
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week),
    is_weekend
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
