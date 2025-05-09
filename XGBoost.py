import mlflow  # type:ignore
import mlflow.xgboost  # type:ignore
import xgboost as xgb  # type:ignore
import pandas as pd  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from sklearn.model_selection import train_test_split  # type:ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score  # type:ignore
from sklearn.preprocessing import LabelEncoder  # type:ignore
from mlflow.models.signature import infer_signature  # type:ignore
import os
import pickle


# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sales_Forecasting_and_Optimization")

# Load and preprocess dataset
df_c = pd.read_csv('global_superstore_2016_cleaned_data.csv')
df_c['Ship Date'] = pd.to_datetime(df_c['Ship Date']).astype('int64') // 10**9
df_c['Order Date'] = pd.to_datetime(df_c['Order Date']).astype('int64') // 10**9

categorical_columns = ['Ship Mode', 'Segment', 'City',
                        'Sub-Category', 'Order Priority',]
label_encoders = {}

for col in categorical_columns:
    if col in df_c.columns:
        le = LabelEncoder()
        df_c[col] = le.fit_transform(df_c[col])
        label_encoders[col] = le

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)


# Define features and target
X = df_c[['Order Date', 'Ship Date', 'Ship Mode', 'Segment', 'City', 'Sub-Category', 
          'Order Priority', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']]
y = df_c['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
y_test_class = (y_test > 0).astype(int)
y_pred_class = (y_pred > 0).astype(int)
accuracy = accuracy_score(y_test_class, y_pred_class)

# Log to MLflow
with mlflow.start_run(run_name="Sales_Forecasting_XGBoost"):
    mlflow.log_metrics({"MAE": mae, "MSE": mse, "R-squared": r2, "Accuracy": accuracy})
    signature = infer_signature(X_train, y_pred)
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model", signature=signature, 
                             registered_model_name="XGBoost_Sales_Forecasting_Model")

    os.makedirs("plots", exist_ok=True)

    # Actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, label='Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('XGBoost - Actual vs Predicted')
    plt.legend()
    plot_path = "plots/actual_vs_predicted.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    residual_plot_path = "plots/residual_plot.png"
    plt.savefig(residual_plot_path)
    mlflow.log_artifact(residual_plot_path)
