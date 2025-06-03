# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
# from datetime import datetime, timedelta
# import logging
# import sys
# import os
# from Scripts.Model_training import *


# sys.path.append('gs://loblaw-bucket/scripts')


# from Scripts.etl import *
# from Scripts.etl_dag import *

# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2025, 3, 16),
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# dag = DAG(
#     'loblaw_etl_and_training',
#     default_args=default_args,
#     description='ETL and model training for Loblaw',
#     schedule_interval=timedelta(days=1),
# )

# def train_models():
#     try:
#         logging.info("Starting model training")
#         train_all_models()  
#         logging.info("Model training completed successfully")
#     except Exception as e:
#         logging.error(f"Error in model training: {e}")
#         raise

# etl_task = PythonOperator(
#     task_id='run_etl',
#     python_callable=run_etl,
#     dag=dag,
# )

# train_model_task = PythonOperator(
#     task_id='train_model',
#     python_callable=main,
#     dag=dag,
# )

# etl_task >> train_model_task
