  output "composer_environment_name" {
  description = "The name of the Cloud Composer environment"
  value       = google_composer_environment.composer_env.name
}

output "composer_environment_dag_gcs_prefix" {
  description = "The Cloud Storage prefix of the DAGs for the Cloud Composer environment"
  value       = google_composer_environment.composer_env.config.0.dag_gcs_prefix
}

output "data_bucket_name" {
  description = "The name of the data bucket"
  value       = google_storage_bucket.data_bucket.name
}
