  
  
  variable "region" {
  description = "The region for resources"
  type        = string
  default     = "us-central1"
}

variable "composer_env_name" {
  description = "The name of the Cloud Composer environment"
  type        = string
  default     = "sales-data-pipeline"
}

variable "bucket_name" {
  description = "The name of the GCS bucket for storing data"
  type        = string
  default     = "loblaw-bucket"
}
