FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn\
    pandas \
    psutil \
    boto3 \
    psycopg2-binary

EXPOSE 5002

CMD ["mlflow"]