from google.cloud import storage
import pickle
import pandas as pd
import io
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/harshsingh/Documents/Loblaw-case-study/key.json"


bucket_name = "loblaw-bucket"
pickle_folder_name = "models"


product_model_files = [
    "27in_4K_Gaming_Monitor_sales_model.pkl",
    "Google_Phone_sales_model.pkl",
    "Macbook_Pro_Laptop_sales_model.pkl",
    "ThinkPad_Laptop_sales_model.pkl",
    "iPhone_sales_model.pkl",
]


client = storage.Client()
bucket = client.get_bucket(bucket_name)

test_data = []

def download_pkl(file_name):
    """Download and load a model from GCS."""
    object_path = f"{pickle_folder_name}/{file_name}"
    blob = bucket.blob(object_path)
    
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    model = pickle.load(buffer)
    
    return model


def evaluate_model(model, test_data, target_col="daily_quantity"):
    feature_cols = ['day_of_week', 'month', 'day', 'lag_1_quantity',
                    'lag_7_quantity', 'lag_30_quantity', 'rolling_7d_mean',
                    'rolling_30d_mean']
    
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# test_data = pd.read_parquet("test_data.parquet")


model_results = {}
for file_name in product_model_files:
    print(f"Evaluating model: {file_name}")
    model = download_pkl(file_name)
    
    evaluation_metrics = evaluate_model(model, test_data)
    model_results[file_name] = evaluation_metrics

for model_name, metrics in model_results.items():
    print(f"{model_name} -> RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.2f}")
