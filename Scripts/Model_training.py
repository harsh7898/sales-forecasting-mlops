#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from google.cloud import storage
import pandas as pd
import numpy as np
import pickle
import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/harshsingh/Documents/Loblaw-case-study/key.json"

bucket_name = 'loblaw-bucket'
prefix = 'raw'
feature_folder_name = 'processed/feature'  
feature_file_name = 'sales-feature.parquet'
pickle_folder_name ='models' 

client = storage.Client()
bucket = client.get_bucket(bucket_name)

def download_blob(folder_name, file_name):
    object_path = f'{folder_name}/{file_name}'
    blob = bucket.blob(object_path)
    
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    
    buffer.seek(0)
    df = pd.read_parquet(buffer)
    return df

def upload_pkl(model, folder_name, file_name):
    object_path = f'{folder_name}/{file_name}'
    blob = bucket.blob(object_path)
    
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    blob.upload_from_file(buffer)

def identify_top_products(df, top_n=5):
    product_sales = df.groupby('product').agg(
        total_value=('total_price', 'sum'),
        total_quantity=('quantity', 'sum')
    ).sort_values('total_value', ascending=False).head(top_n)
    
    return product_sales.index.tolist()

def prepare_product_features(df, product_name):
    product_df = df[df['product'] == product_name].copy()
    return product_df

def time_series_train_test_split(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test

def build_and_evaluate_model(train_data, test_data, target_col='daily_quantity'):
    feature_cols = ['day_of_week', 'month', 'day', 'lag_1_quantity', 
                    'lag_7_quantity', 'lag_30_quantity', 'rolling_7d_mean', 
                    'rolling_30d_mean']
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_score:
            best_score = r2
            best_model = model
    
    return best_model

def predict_sales_for_top_products(df, top_products=None):
    if top_products is None:
        top_products = identify_top_products(df)
    
    product_models = {}
    
    for product in top_products:
        print(f"Building model for: {product}")
        
        product_data = prepare_product_features(df, product)
        train_data, test_data = time_series_train_test_split(product_data)
        best_model = build_and_evaluate_model(train_data, test_data)
        
        file_name= f"{product.replace(' ', '_')}_sales_model.pkl"
        upload_pkl(best_model, folder_name=pickle_folder_name, file_name=file_name)
        
        product_models[product] = {'model': best_model}
    
    return product_models


def main():

    download_df = download_blob(feature_folder_name, feature_file_name)
    product_models = predict_sales_for_top_products(download_df, identify_top_products(download_df))

if __name__ == "__main__":
    main()