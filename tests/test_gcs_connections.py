from google.cloud import storage
import os
import pytest

def test_gcs_connection():
    key_path = os.environ.get("GCS_KEY_PATH")
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "loblaw-bucket")
    
    assert key_path is not None, "GCS_KEY_PATH environment variable is not set"
    assert os.path.exists(key_path), f"Key file not found at {key_path}"
    
    client = storage.Client.from_service_account_json(key_path)
    
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs())
    
    # Assert tat we can list blobs, indicating successful connection
    assert blobs is not None, "No blobs found in the bucket, connection may have failed"
  
  
  
  