
# Sales Forecasting with MLOps

This project is an end-to-end machine learning pipeline designed to predict the sales quantity for the top 5 selling products in a large retail dataset (2019), excluding books and liquor. The pipeline follows best practices in ETL, business analytics, machine learning modeling, scalability, and MLOps automation using Docker, Airflow, and GCP Composer.
## Features

- ETL pipeline for large-scale retail sales data

-  Data cleaning, transformation, and parquet-based storage
-  Business insights for stakeholders with clear visuals
-  Product-wise sales forecasting using machine learning models
-  Scalable training pipelines using Dockerized Airflow
-  GCP deployment with Composer, Cloud Storage, Vertex AI (via Terraform)
-  CI/CD via GitHub Actions

## Documentation:

[Dataset](https://www.kaggle.com/datasets/pigment/big-sales-data/data) - Kaggle Dataset

Notebooks folder contains all the jupyter file used for development:

File Name:
etl.ipynb, Model_training.ipynb, evaluation.ipynb, analysis.ipynb (It includes Business Insights)

/Scripts folder: Contains all the development Script files

/airflow-docker/dags/Scripts folder: Contains the main dag and final Scripts for pipeline

/findings folder: Contains business analysis reports.

/images folder: Contains all the images of successful final run

main.tf, variables.tf, outputs.tf are terraform files which allows to define and manage cloud infrastructure.

/.github/workflows: Contains deploy_to_gcp.yml file for github actions.

/tests: Contains test files
## Machine Learning Models

- Linear Regression
- Random Forest
- Gradient Boosting
## Scalability and MLOps

- Local Automation (Airflow)
- Developed ETL + training pipelines as Airflow DAGs.
- Dockerized environment in airflow-docker/
- Models trained in parallel with separate tasks.
## Cloud Automation (GCP)

Used GCP Composer + Terraform + GitHub Actions to deploy pipeline:
- Infrastructure setup via main.tf, variables.tf
- Trigger via deploy_to_gcp.yml


##  Run Locally

1. Clone the repo:

```bash
  git clone https://github.com/harsh7898/sales-forecasting-mlops
  cd sales-forecasting-mlops
```

2. Set up virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run Airflow locally:

```bash
cd airflow-docker/
docker-compose up
```

4. Access Airflow UI and trigger DAGs.


## Run on GCP

1. Initialize Terraform:

```bash
terraform init
terraform apply
```

2. Push code to GitHub -> CI/CD triggers GCP Composer.
## Resources and References

[Pandas Documentation](https://pandas.pydata.org/docs/)

[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

[Airflow Docs](https://airflow.apache.org/docs/)

[Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

[GCP Composer Setup](https://cloud.google.com/composer/docs)