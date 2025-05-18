# Sales-Forecasting-and-Optimization
## Project Overview
Accurate sales forecasts are vital for any company that wants to:

- **Plan inventory** more efficiently and avoid stockouts or overstock  
- **Align marketing campaigns** with expected demand peaks  
- **Minimize losses** from unsold products or rushed stock orders

In this repo, you’ll find everything from raw data to a live, user‑friendly dashboard—plus the code and pipelines that tie it all together.



### 1. `data`
- **global_superstore_2016.xlsx**: Our raw dataset, sourced from Global Superstore 2016.  
- **global_superstore_2016_cleaned_data.csv**: The cleaned, preprocessed version that feeds into our models.

### 2. `notebooks`
- **Exploratory Data Analysis (EDA) Notebook.ipynb**: Step‑by‑step data cleaning, feature engineering, and visualization to understand sales patterns.  
- **Final_Project.ipynb**: Model training and evaluation, including comparisons between multiple time‑series approaches and the selection of XGBoost.

### 3. `powerbi`
- **Final Project Sales (Power BI).rar**: A set of interactive Power BI dashboards showcasing key KPIs, trend analyses, and forecasting results.

### 4. `models`
- **label_encoders.pkl**: Serialized `LabelEncoder` objects used to transform categorical features during modeling. These are reloaded in the Streamlit app to decode user inputs back to their original categories.

### 5. `src`
- **XGBoost.py**: Script to train the XGBoost model, log parameters and metrics to MLflow, and register new model versions.  
- **PREDICTION.py**: Streamlit application that allows users to enter new data, get instant forecasts, and trigger email alerts when predictions deviate beyond a set threshold.

## Explore the data
- Open notebooks/Exploratory Data Analysis (EDA) Notebook.ipynb to see how we cleaned and visualized the raw data.
- Check out data/global_superstore_2016_cleaned_data.csv for the finalized dataset.

## How It Works
- Data Cleaning & EDA: We handle missing values, encode categories, and visualize sales over time.
- Modeling: Multiple time‑series approaches are tested; XGBoost (with date as index) gives the best accuracy.
- Tracking with MLflow: Every run logs parameters, metrics, and artifacts so you can compare versions at a glance.
- Interactive UI: A Streamlit front‑end lets users input new sales figures, view predictions, and submit actuals back into the pipeline.
- Alerts & Feedback Loop: If the prediction error exceeds our threshold, an automated email notifies the responsible team member—and the new actuals can be appended to retrain the model.

## Acknowledgments
Thank you to Mostafa Elshahat, Mohab Mohamed, Ziad Ahmed Gharieb for making this project possible, and to Eyouth for organizing the DEPI initiative.

Feel free to open an issue or submit a pull request. Any suggestions for new features, improved documentation, or bug fixes are welcome!
