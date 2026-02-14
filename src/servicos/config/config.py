import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()


class Config:
    STEAM_API_URL: Final[str] = os.getenv("STEAM_API_URL", " ")
    MINIO_ENDPOINT: Final[str] = os.getenv("MLFLOW_S3_ENDPOINT_URL", " ")
    MINIO_ACCESS_KEY: Final[str] = os.getenv("MINIO_ROOT_USER", " ")
    MINIO_SECRET_KEY: Final[str] = os.getenv("MINIO_ROOT_PASSWORD", " ")
    MINIO_BUCKET: Final[str] = os.getenv("MINIO_BUCKET")
    AWS_REGION: Final[str] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
