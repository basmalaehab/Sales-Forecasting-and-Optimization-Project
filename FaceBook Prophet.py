import mlflow  # type:ignore
import mlflow.prophet  # type:ignore
import pandas as pd  # type:ignore
import numpy as np  # type:ignore
from prophet import Prophet  # type:ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from mlflow.models.signature import infer_signature  # type:ignore
import os

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sales_Forecasting_and_Optimization")

# Evaluate metrics function
def evaluate_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

# Load the dataset
df = pd.read_csv('global_superstore_2016_cleaned_data.csv')

# Rename columns to match Prophet's requirements
df_for_prophet = df.reset_index()[['Order Date', 'Sales']].rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Initialize Prophet model
model = Prophet()

# Fit the model
model.fit(df_for_prophet)

# Make future dataframe and forecast
future = model.make_future_dataframe(df_for_prophet, periods=30)  # Forecasting 30 days into the future
forecast = model.predict(future)

# Merge forecast with actuals for evaluation (Ensure 'ds' columns are datetime in both)
df_for_prophet['ds'] = pd.to_datetime(df_for_prophet['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])
merged = pd.merge(forecast[['ds', 'yhat']], df_for_prophet[['ds', 'y']], on='ds', how='inner')

# Calculate metrics
mse, mae, r2 = evaluate_metrics(merged['y'], merged['yhat'])

# Calculate accuracy (since Prophet is a regression model, accuracy calculation is based on an arbitrary threshold)
accuracy = accuracy_score((merged['y'] > 0).astype(int), (merged['yhat'] > 0).astype(int))

# Log metrics and model in MLflow
with mlflow.start_run(run_name="Sales_Forecasting_Facebook_Prophet"):
    # Log metrics
    mlflow.log_metrics({"MSE": mse, "MAE": mae, "R-squared": r2, "Accuracy": accuracy})

    # Create the signature for the model
    signature = infer_signature(df_for_prophet[['ds', 'y']], forecast[['ds', 'yhat']])

    # Log Facebook Prophet model with the signature
    mlflow.prophet.log_model(
        model,
        artifact_path="prophet_model",
        signature=signature,
        registered_model_name="Prophet_Sales_Forecasting_Model"
    )

    # Create directory for plots if not exists
    os.makedirs("plots", exist_ok=True)

    # Log the forecast plot
    fig = model.plot(forecast)
    forecast_plot_path = "plots/forecast_plot.png"
    fig.savefig(forecast_plot_path)
    mlflow.log_artifact(forecast_plot_path)

    # Log the forecast components
    fig_components = model.plot_components(forecast)
    components_plot_path = "plots/components_plot.png"
    fig_components.savefig(components_plot_path)
    mlflow.log_artifact(components_plot_path)
