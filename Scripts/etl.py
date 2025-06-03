import warnings
warnings.filterwarnings("ignore")
from google.cloud import storage
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re
import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/harshsingh/Documents/Loblaw-case-study/key.json"

bucket_name = 'loblaw-bucket'
prefix = 'raw'
transformed_folder_name = 'processed/transformed'
transformed_file_name = 'sales-processed.parquet'
feature_folder_name = 'processed/feature'
feature_file_name = 'sales-feature.parquet'

client = storage.Client()
bucket = client.get_bucket(bucket_name)

def raw_data():
    blobs = bucket.list_blobs(prefix=prefix)
    csv_files = [blob.name for blob in blobs if blob.name.endswith('.csv')]
    df_list = []
    for file_name in csv_files:
        blob = bucket.blob(file_name)
        file_content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(file_content))
        df_list.append(df)
    raw_df = pd.concat(df_list, ignore_index=True)
    return raw_df

def upload_to_blob_as_parquet(df, folder_name, file_name):
    bucket = client.bucket(bucket_name)
    object_path = f'{folder_name}/{file_name}'
    blob = bucket.blob(object_path)
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine='pyarrow')
    buffer.seek(0)
    blob.upload_from_file(buffer)

def download_blob(folder_name, file_name):    
    object_path = f'{folder_name}/{file_name}'
    blob = bucket.blob(object_path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    df = pd.read_parquet(buffer)
    return df

def clean_sales_data(df):
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('')
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(0)
    if 'Order ID' in df.columns:
        df = df.rename(columns={
            'Order ID': 'order_id',
            'Product': 'product',
            'Quantity Ordered': 'quantity',
            'Price Each': 'each_price',
            'Order Date': 'order_date',
            'Purchase Address': 'purchased_address'
        })
    if 'order_date' in df.columns:
        df['order_date'] = df['order_date'].astype(str).str.strip()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['month'] = np.where(df['order_date'].isna(), 0, df['order_date'].dt.month).astype(int)
        df['day_of_week'] = df['order_date'].dt.day_name()
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['each_price'] = pd.to_numeric(df['each_price'], errors='coerce').fillna(0).astype(float)
    df['total_price'] = df['quantity'] * df['each_price']
    product_mapping = {product: idx + 1 for idx, product in enumerate(df['product'].unique())}
    df['product_id'] = df['product'].map(product_mapping)
    address_col = 'purchased_address' if 'purchased_address' in df.columns else 'Purchase Address'
    if address_col in df.columns:
        df[address_col] = df[address_col].fillna('')
        df['street'] = ''
        df['city'] = ''
        df['state'] = ''
        df['zip'] = ''
        for i, row in df.iterrows():
            addr = row[address_col]
            if addr and isinstance(addr, str):
                parts = [part.strip() for part in addr.split(',')]
                if len(parts) >= 3:
                    df.at[i, 'street'] = parts[0]
                    df.at[i, 'city'] = parts[1]
                    state_zip = parts[2].strip()
                    match = re.search(r'([A-Z]{2})\s+(\d{5})', state_zip)
                    if match:
                        df.at[i, 'state'] = match.group(1)
                        df.at[i, 'zip'] = match.group(2)
    return df


# def feature_extraction(df):

#     drop_cols = ['purchased_address','order_id','zip','street','city']
#     df['quarter'] = df['order_date'].dt.quarter
#     df['dayofmonth'] = df['order_date'].dt.day
#     df['weekofyear'] = df['order_date'].dt.isocalendar().week
    

#     df = df.sort_values('order_date')
#     df['prev_day_qty'] = df['quantity'].shift(1)
#     df['prev_week_qty'] = df['quantity'].shift(7)
#     df['prev_month_qty'] = df['quantity'].shift(30)

#     df['rolling_7d_avg'] = df['quantity'].rolling(window=7).mean()
#     df['rolling_30d_avg'] = df['quantity'].rolling(window=30).mean()


#      # Group by date to get daily quantity
#     daily_data = product_df.groupby('order_date').agg(
#         daily_quantity=('quantity', 'sum'),
#         daily_revenue=('total_price', 'sum'),
#         avg_price=('each_price', 'mean')
#     ).reset_index()
    
#     # Add day of week, month features
#     daily_data['day_of_week'] = daily_data['order_date'].dt.dayofweek
#     daily_data['month'] = daily_data['order_date'].dt.month
#     daily_data['day'] = daily_data['order_date'].dt.day
    
#     # Sort by date
#     daily_data = daily_data.sort_values('order_date')
    
#     # Add lag features (previous day's sales)
#     daily_data['lag_1_quantity'] = daily_data['daily_quantity'].shift(1)
#     daily_data['lag_7_quantity'] = daily_data['daily_quantity'].shift(7)  # 1 week ago
#     daily_data['lag_30_quantity'] = daily_data['daily_quantity'].shift(30)  # 1 month ago
    
#     # Add rolling window features
#     daily_data['rolling_7d_mean'] = daily_data['daily_quantity'].rolling(window=7).mean()
#     daily_data['rolling_30d_mean'] = daily_data['daily_quantity'].rolling(window=30).mean()
    

#     df = df.dropna(ignore_index= True)
    
#     df = df.drop(drop_cols, axis=1)

#     return df


def feature_extraction(df):
    drop_cols = ['purchased_address', 'order_id', 'zip', 'street', 'city']
    

    df['quarter'] = df['order_date'].dt.quarter
    df['dayofmonth'] = df['order_date'].dt.day
    df['weekofyear'] = df['order_date'].dt.isocalendar().week

    df = df.sort_values(['product_id', 'order_date'])
    

    df['prev_day_qty'] = df.groupby('product_id')['quantity'].shift(1)
    df['prev_week_qty'] = df.groupby('product_id')['quantity'].shift(7)
    df['prev_month_qty'] = df.groupby('product_id')['quantity'].shift(30)
    
    # df['rolling_7d_avg'] = df.groupby('product_id')['quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    # df['rolling_30d_avg'] = df.groupby('product_id')['quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
    

    df['daily_quantity'] = df.groupby(['product_id', 'order_date'])['quantity'].transform('sum')
    df['daily_revenue'] = df.groupby(['product_id', 'order_date'])['total_price'].transform('sum')
    df['avg_price'] = df.groupby(['product_id', 'order_date'])['each_price'].transform('mean')
    

    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    

    df['lag_1_quantity'] = df.groupby('product_id')['daily_quantity'].shift(1)
    df['lag_7_quantity'] = df.groupby('product_id')['daily_quantity'].shift(7)
    df['lag_30_quantity'] = df.groupby('product_id')['daily_quantity'].shift(30)
    

    df['rolling_7d_mean'] = df.groupby('product_id')['daily_quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    df['rolling_30d_mean'] = df.groupby('product_id')['daily_quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
    
    df = df.drop(columns=drop_cols, errors='ignore').dropna().reset_index(drop=True)
    
    return df


def main():
    
    raw_df = raw_data()
    cured_df = clean_sales_data(raw_df)
    upload_to_blob_as_parquet(cured_df, transformed_folder_name, transformed_file_name)
    featured_df = feature_extraction(cured_df)
    upload_to_blob_as_parquet(featured_df, feature_folder_name, feature_file_name)

if __name__ == "__main__":
    main()