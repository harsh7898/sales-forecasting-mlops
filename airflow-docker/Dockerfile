FROM apache/airflow:2.10.5

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
COPY key.json /opt/airflow/key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/key.json

