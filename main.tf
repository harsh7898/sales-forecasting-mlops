  provider "google" {
  project = "loblaw-sales-project"
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "composer_api" {
  project            = "loblaw-sales-project"
  service            = "composer.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
  project            = "loblaw-sales-project"
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  project            = "loblaw-sales-project"
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# Create service account for Composer
resource "google_service_account" "composer_service_account" {
  account_id   = "composer-sa"
  display_name = "Service Account for Cloud Composer"
}

# Grant necessary permissions to service account
resource "google_project_iam_member" "composer_worker" {
  project = "loblaw-sales-project"
  role    = "roles/composer.worker"
  member  = "serviceAccount:${google_service_account.composer_service_account.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = "loblaw-sales-project"
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.composer_service_account.email}"
}

resource "google_project_iam_member" "storage_object_admin" {
  project = "loblaw-sales-project"
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.composer_service_account.email}"
}

# Create storage bucket for data
resource "google_storage_bucket" "data_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
}

# # Create folder structure in the bucket
# resource "google_storage_bucket_object" "transformed_folder" {
#   name    = "processed/transformed/"
#   content = ""
#   bucket  = google_storage_bucket.data_bucket.name
#   depends_on = [google_storage_bucket.data_bucket]
# }

# resource "google_storage_bucket_object" "feature_folder" {
#   name    = "processed/feature/"
#   content = ""
#   bucket  = google_storage_bucket.data_bucket.name
#   depends_on = [google_storage_bucket.data_bucket]
# }

# resource "google_storage_bucket_object" "models_folder" {
#   name    = "models/"
#   content = ""
#   bucket  = google_storage_bucket.data_bucket.name
#   depends_on = [google_storage_bucket.data_bucket]
# }

# Create Cloud Composer environment
resource "google_composer_environment" "composer_env" {
  name   = var.composer_env_name
  region = var.region

  depends_on = [
    google_project_service.composer_api,
    google_project_service.compute_api,
    google_project_service.storage_api,
  ]

  config {
    node_config {
      service_account = google_service_account.composer_service_account.email
    }

    software_config {
      image_version = "composer-3-airflow-2.10.2"

      pypi_packages = {
        "pandas" = "==2.1.4"
        "numpy" = "==1.26.4"
        "scikit-learn" = "==1.2.2"
        "xgboost" = "==1.7.5"
        "google-cloud-storage" = "==2.9.0"
        "pyarrow" = "==16.1.0"
        # "fastparquet" = "==2023.4.0"
      }

      env_variables = {
        "AIRFLOW_VAR_GCP_PROJECT_ID"     = "loblaw-sales-project"
        "AIRFLOW_VAR_BUCKET_NAME"        = google_storage_bucket.data_bucket.name
        "AIRFLOW_VAR_TRANSFORMED_FOLDER" = "processed/transformed"
        "AIRFLOW_VAR_FEATURE_FOLDER"     = "processed/feature"
        "AIRFLOW_VAR_MODELS_FOLDER"      = "models"
      }
    }
  }
}

# # Create a folder for DAG scripts
# resource "google_storage_bucket_object" "scripts_folder" {
#   name    = "dags/Scripts/"
#   content = ""
#   bucket  = google_storage_bucket.data_bucket.name
#   depends_on = [google_composer_environment.composer_env]
# }
