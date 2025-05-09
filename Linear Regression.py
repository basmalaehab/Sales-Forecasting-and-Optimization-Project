# Linear Regression with MLflow tracking and model signature
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sales_Forecasting_and_Optimization")

# Load dataset
df_c = pd.read_csv("global_superstore_2016_cleaned_data.csv")

# Start with the target variable
target = 'Profit'
if target in df_c.columns:
    X = df_c.select_dtypes(include=['number']).drop(columns=[target])
    y = df_c[target]
    X, y = X.align(y, join='inner', axis=0)
else:
    raise KeyError(f"'{target}' column not found in the DataFrame.")

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index.intersection(y.index)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLflow experiment tracking
with mlflow.start_run(run_name="Linear_Regression_Model"):
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    # Log metrics
    mlflow.log_metrics({"MAE": mae, "MSE": mse, "R-squared": r2})


    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig("linear_regression_plot.png")
    mlflow.log_artifact("linear_regression_plot.png")
    plt.show()

    # Infer and log model signature
    signature = infer_signature(X_test_scaled, y_pred)

    # Log model with signature
    mlflow.sklearn.log_model(model, artifact_path= "linear_regression_model", signature=signature, registered_model_name="Linear_Regression_Sales_Model")