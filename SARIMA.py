import mlflow  # type:ignore
import mlflow.statsmodels  # type:ignore
import pandas as pd  # type:ignore
import numpy as np  # type:ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type:ignore
#import matplotlib.pyplot as plt  # type:ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type:ignore
from mlflow.models.signature import infer_signature  # type:ignore


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

# Fit SARIMA model
sarima_model = SARIMAX(df_c['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()
print(sarima_model_fit.summary())

# Forecasting with SARIMA
forecast_sarima = sarima_model_fit.forecast(steps=12)  # Forecast for the next 12 months
print(forecast_sarima)

# Calculate evaluation metrics
mse, mae, r2 = evaluate_metrics(df_c['Sales'][-12:], forecast_sarima)

# Log metrics and model in MLflow
with mlflow.start_run(run_name="Sales_Forecasting_SARIMA"):
    # Log metrics
    mlflow.log_metrics({"MSE": mse, "MAE": mae, "R-squared": r2})

    # Create the signature for the model
    signature = infer_signature(df_c[['Sales']], forecast_sarima)

    # Log SARIMA model with the signature
    mlflow.statsmodels.log_model(sarima_model_fit, artifact_path="sarima_model", signature=signature, registered_model_name="SARIMA_Sales_Forecasting_Model")

    # Log the forecast plot
    #plt.figure(figsize=(12, 6))
    #plt.plot(df_c['Sales'], label='Historical Sales')
    #plt.plot(range(len(df_c['Sales']), len(df_c['Sales']) + len(forecast_sarima)), forecast_sarima, label='Forecasted Sales', color='red')
    #plt.title('Sales Forecast using SARIMA')
    #plt.xlabel('Index')
    #plt.ylabel('Sales')
    #plt.legend()
    #forecast_plot_path = "plots/forecast_plot.png"
    #plt.savefig(forecast_plot_path)
    #mlflow.log_artifact(forecast_plot_path)
    #plt.close()

    # Skip logging ACF/PACF plots to save space
    # Remove these lines if you don't want to log ACF/PACF plots:
    # ACF Plot
    # plt.figure()
    # plot_acf(df_c['Sales'], lags=40)
    # acf_plot_path = "plots/acf_plot.png"
    # plt.savefig(acf_plot_path)
    # mlflow.log_artifact(acf_plot_path)
    # plt.close()

    # PACF Plot
    # plt.figure()
    # plot_pacf(df_c['Sales'], lags=40)
    # pacf_plot_path = "plots/pacf_plot.png"
    # plt.savefig(pacf_plot_path)
    # mlflow.log_artifact(pacf_plot_path)
    # plt.close()
