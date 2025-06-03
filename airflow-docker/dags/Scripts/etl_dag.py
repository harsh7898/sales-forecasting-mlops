# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
# from datetime import datetime, timedelta
# import logging
# from Scripts.etl import *

# default_args = {
#     'owner': 'airflow',
#     'start_date': datetime(2025, 3, 16),
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# dag = DAG('etl_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

# def run_etl():
#     # ETL function
#     try:
#         logging.info("Starting ETL process")
#         from etl import run_etl_process
#         run_etl_process()
#         logging.info("ETL process completed successfully")
#     except Exception as e:
#         logging.error(f"Error in ETL process: {e}")
#         raise

# etl_task = PythonOperator(
#     task_id='run_etl',
#     python_callable= main,
#     dag=dag,
# )









# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
# from datetime import datetime, timedelta
# import logging
# import os
# import sys
# from Scripts.etl import *  

# default_args = {
#     'owner': 'airflow',
#     'start_date': datetime(2025, 3, 16),
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# dag = DAG('etl_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

# def run_etl():
#     try:
#         logging.info("Starting ETL process")
#         run_etl_process() 
#         logging.info("ETL process completed successfully")
#     except Exception as e:
#         logging.error(f"Error in ETL process: {e}")
#         raise

# etl_task = PythonOperator(
#     task_id='run_etl',
#     python_callable=run_etl, 
#     dag=dag,
# )
