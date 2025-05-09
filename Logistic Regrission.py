# Logistic Regression with MLflow tracking and model signature
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from mlflow.models.signature import infer_signature

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sales_Forecasting_and_Optimization")

# Load dataset
df_c = pd.read_csv("global_superstore_2016_cleaned_data.csv")

# Define target and features
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

# Convert target to binary (classification)
y_binary = (y > 0).astype(int)

# Train-test split (same as linear)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLflow tracking
with mlflow.start_run(run_name="Logistic_Regression_Model"):
    # Create and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:\n", cm)

    # Log metrics
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_logistic.png")
    mlflow.log_artifact("confusion_matrix_logistic.png")
    plt.show()

    # Infer and log model signature
    signature = infer_signature(X_test_scaled, y_pred)

    # Log model
    mlflow.sklearn.log_model(
        model, 
        artifact_path="logistic_regression_model", 
        signature=signature, 
        registered_model_name="Logistic_Regression_Sales_Model")
