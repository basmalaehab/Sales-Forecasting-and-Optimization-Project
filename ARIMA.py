import mlflow  #type:ignore
import mlflow.statsmodels #type:ignore
import pandas as pd #type:ignore
import numpy as np #type:ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  #type:ignore
import matplotlib.pyplot as plt #type:ignore
from statsmodels.tsa.arima.model import ARIMA #type:ignore
from statsmodels.tsa.stattools import adfuller #type:ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #type:ignore
from mlflow.models.signature import infer_signature #type:ignore
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

# Example dataset (use your own 'df' or 'df_c' here)
df_c = pd.read_csv("global_superstore_2016_cleaned_data.csv")

# Handle infinite and NaN values
df_c['Sales'] = df_c['Sales'].replace([np.inf, -np.inf], np.nan)
df_c = df_c.dropna(subset=['Sales'])

# Fit ARIMA model
model = ARIMA(df_c['Sales'], order=(1, 1, 1))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=12)

# Calculate evaluation metrics
mse, mae, r2 = evaluate_metrics(df_c['Sales'][-12:], forecast)

# Log metrics and model in MLflow
with mlflow.start_run(run_name="Sales_Forecasting_ARIMA"):
    # Log metrics
    mlflow.log_metrics({"MSE": mse, "MAE": mae, "R-squared": r2})

    # Log ARIMA model
    signature = infer_signature(df_c[['Sales']], forecast)
    mlflow.statsmodels.log_model(model_fit, artifact_path="arima_model", signature=signature, registered_model_name="ARIMA_Sales_Forecasting_Model")

    # Create folder to store plots
    os.makedirs("plots", exist_ok=True)

    # Log the forecast plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_c['Sales'], label='Historical Sales')
    plt.plot(range(len(df_c['Sales']), len(df_c['Sales']) + len(forecast)), forecast, label='Forecasted Sales', color='red')
    plt.title('Sales Forecast using ARIMA')
    plt.xlabel('Index')
    plt.ylabel('Sales')
    plt.legend()
    forecast_plot_path = "plots/forecast_plot.png"
    plt.savefig(forecast_plot_path)
    mlflow.log_artifact(forecast_plot_path)
    plt.close()

    # ACF Plot
    plt.figure()
    plot_acf(df_c['Sales'], lags=40)
    acf_plot_path = "plots/acf_plot.png"
    plt.savefig(acf_plot_path)
    mlflow.log_artifact(acf_plot_path)
    plt.close()

    # PACF Plot
    plt.figure()
    plot_pacf(df_c['Sales'], lags=40)
    pacf_plot_path = "plots/pacf_plot.png"
    plt.savefig(pacf_plot_path)
    mlflow.log_artifact(pacf_plot_path)
    plt.close()
