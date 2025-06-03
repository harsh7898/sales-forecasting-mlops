from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from Scripts.etl import raw_data, clean_sales_data, feature_extraction, upload_to_blob_as_parquet
    from Scripts.Model_training import identify_top_products, prepare_product_features, time_series_train_test_split, build_and_evaluate_model, predict_sales_for_top_products, download_blob, upload_pkl
    from Scripts.evaluate import *             
    logger.info("Successfully imported functions from etl.py and model.py")
except ImportError as e:
    logger.error(f"Error importing functions: {e}")
    raise

BUCKET_NAME = 'loblaw-bucket'
TRANSFORMED_FOLDER_NAME = 'processed/transformed'
TRANSFORMED_FILE_NAME = 'sales-processed.parquet'
FEATURE_FOLDER_NAME = 'processed/feature'
FEATURE_FILE_NAME = 'sales-feature.parquet'
PICKLE_FOLDER_NAME = 'models'

def extract_raw_data(**kwargs):
    logger.info("Starting raw data extraction")
    try:
        raw_df = raw_data()
        logger.info(f"Raw data extracted. Shape: {raw_df.shape}")
        
        ti = kwargs['ti']
        ti.xcom_push(key='raw_data_shape', value=raw_df.shape)
        
        return raw_df.to_json()
    except Exception as e:
        logger.error(f"Error in raw data extraction: {e}")
        raise

def transform_raw_data(**kwargs):
    logger.info("Starting data transformation")
    try:
        ti = kwargs['ti']
        raw_data_json = ti.xcom_pull(task_ids='extract_raw_data')
        
        import pandas as pd
        import io
        raw_df = pd.read_json(raw_data_json)
        logger.info(f"Raw data loaded from XCom. Shape: {raw_df.shape}")
        
        cured_df = clean_sales_data(raw_df)
        logger.info(f"Data cleaned. Shape: {cured_df.shape}")
        
        upload_to_blob_as_parquet(cured_df, TRANSFORMED_FOLDER_NAME, TRANSFORMED_FILE_NAME)
        logger.info(f"Transformed data uploaded to {TRANSFORMED_FOLDER_NAME}/{TRANSFORMED_FILE_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"Error in data transformation process: {e}")
        raise

def etl_transformed_to_features():
    logger.info("Starting ETL process: transformed to features")
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        
        object_path = f'{TRANSFORMED_FOLDER_NAME}/{TRANSFORMED_FILE_NAME}'
        blob = bucket.blob(object_path)
        
        import io
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        
        buffer.seek(0)
        import pandas as pd
        cured_df = pd.read_parquet(buffer)
        
        logger.info(f"Transformed data downloaded. Shape: {cured_df.shape}")
        
        featured_df = feature_extraction(cured_df)
        logger.info(f"Features extracted. Shape: {featured_df.shape}")
        
        upload_to_blob_as_parquet(featured_df, FEATURE_FOLDER_NAME, FEATURE_FILE_NAME)
        logger.info(f"Feature data uploaded to {FEATURE_FOLDER_NAME}/{FEATURE_FILE_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"Error in feature extraction process: {e}")
        raise

def train_model_for_product(product_name, **kwargs):
    logger.info(f"Starting model training for product: {product_name}")
    try:
        featured_df = download_blob(FEATURE_FOLDER_NAME, FEATURE_FILE_NAME)
        logger.info(f"Feature data downloaded. Shape: {featured_df.shape}")
        
        product_data = prepare_product_features(featured_df, product_name)
        logger.info(f"Product data prepared. Shape: {product_data.shape}")
        
        train_data, test_data = time_series_train_test_split(product_data)
        logger.info(f"Data split. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        
        best_model = build_and_evaluate_model(train_data, test_data)
        logger.info(f"Model built and evaluated for product: {product_name}")
        
        file_name = f"{product_name.replace(' ', '_')}_sales_model.pkl"
        upload_pkl(best_model, folder_name=PICKLE_FOLDER_NAME, file_name=file_name)
        logger.info(f"Model saved to {PICKLE_FOLDER_NAME}/{file_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error in model training for product {product_name}: {e}")
        raise

def identify_top_products_task(**kwargs):
    logger.info("Starting to identify top products")
    try:
        featured_df = download_blob(FEATURE_FOLDER_NAME, FEATURE_FILE_NAME)
        logger.info(f"Feature data downloaded. Shape: {featured_df.shape}")
        
        top_products = identify_top_products(featured_df)
        logger.info(f"Top products identified: {top_products}")
        
        ti = kwargs['ti']
        ti.xcom_push(key='top_products', value=top_products)
        
        return top_products
    except Exception as e:
        logger.error(f"Error identifying top products: {e}")
        raise

def create_model_training_tasks(top_product_task_id, dag):
    def get_product_training_callback(product_idx):
        def callback(**kwargs):
            ti = kwargs['ti']
            top_products = ti.xcom_pull(task_ids=top_product_task_id, key='top_products')
            if product_idx < len(top_products):
                return train_model_for_product(top_products[product_idx], **kwargs)
            else:
                logger.warning(f"Product index {product_idx} out of range. Only {len(top_products)} products available.")
                return None
        return callback
    
    training_tasks = []
    for i in range(5):
        task = PythonOperator(
            task_id=f'train_model_for_product_{i}',
            python_callable=get_product_training_callback(i),
            provide_context=True,
            dag=dag,
        )
        training_tasks.append(task)
    
    return training_tasks

def evaluate_all_models(**kwargs):
    logger.info("Starting model evaluation")
    try:
        from google.cloud import storage
        import pickle
        import pandas as pd
        import io
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        
        ti = kwargs['ti']
        top_products = ti.xcom_pull(task_ids='identify_top_products', key='top_products')
        
        if not top_products:
            logger.warning("No top products found. Using predefined list.")
            product_model_files = [
                "27in_4K_Gaming_Monitor_sales_model.pkl",
                "Google_Phone_sales_model.pkl",
                "Macbook_Pro_Laptop_sales_model.pkl",
                "ThinkPad_Laptop_sales_model.pkl",
                "iPhone_sales_model.pkl",
            ]
        else:
            product_model_files = [f"{product.replace(' ', '_')}_sales_model.pkl" for product in top_products[:5]]
        
        logger.info(f"Models to evaluate: {product_model_files}")
        
        featured_df = download_blob(FEATURE_FOLDER_NAME, FEATURE_FILE_NAME)
        logger.info(f"Feature data downloaded for evaluation. Shape: {featured_df.shape}")
        
        def download_pkl(file_name):
            object_path = f"{PICKLE_FOLDER_NAME}/{file_name}"
            blob = bucket.blob(object_path)
            
            if not blob.exists():
                logger.warning(f"Model file {file_name} does not exist")
                return None
                
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
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {"RMSE": rmse, "MAE": mae, "R2": r2}
        
        model_results = {}
        for file_name in product_model_files:
            product_name = file_name.replace('_sales_model.pkl', '').replace('_', ' ')
            logger.info(f"Evaluating model for product: {product_name}")
            
            model = download_pkl(file_name)
            if model is None:
                logger.warning(f"Could not load model for {product_name}, skipping evaluation")
                continue
                
            product_data = prepare_product_features(featured_df, product_name)
            if product_data.empty:
                logger.warning(f"No data found for product {product_name}, skipping evaluation")
                continue
                
            _, test_data = time_series_train_test_split(product_data)
            
            try:
                evaluation_metrics = evaluate_model(model, test_data)
                model_results[product_name] = evaluation_metrics
                logger.info(f"{product_name} -> RMSE: {evaluation_metrics['RMSE']:.2f}, MAE: {evaluation_metrics['MAE']:.2f}, R2: {evaluation_metrics['R2']:.2f}")
            except Exception as e:
                logger.error(f"Error evaluating model for {product_name}: {e}")
        
        return model_results
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'parallel_etl_model_training',
    default_args=default_args,
    description='Parallel ETL and Model Training DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:
    
    start = DummyOperator(
        task_id='start',
    )
    
    extract_raw = PythonOperator(
        task_id='extract_raw_data',
        python_callable=extract_raw_data,
        provide_context=True,
    )
    
    transform_raw = PythonOperator(
        task_id='transform_raw_data',
        python_callable=transform_raw_data,
        provide_context=True,
    )
    
    transformed_to_features = PythonOperator(
        task_id='transformed_to_features',
        python_callable=etl_transformed_to_features,
    )
    
    identify_products = PythonOperator(
        task_id='identify_top_products',
        python_callable=identify_top_products_task,
        provide_context=True,
    )
    
    training_tasks = create_model_training_tasks('identify_top_products', dag)
    
    evaluate_models = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_all_models,
        provide_context=True,
    )
    
    end = DummyOperator(
        task_id='end',
    )
    
    start >> extract_raw >> transform_raw >> transformed_to_features >> identify_products
    
    for task in training_tasks:
        identify_products >> task >> evaluate_models
        
    evaluate_models >> end

if __name__ == "__main__":
    dag.cli()
