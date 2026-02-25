import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils.utils import configurar_mlflow

MLFLOW_URI = "http://172.26.0.5:5000"

EXPERIMENT_NAME = f"exemplo dois"
configurar_mlflow(experiment_name=EXPERIMENT_NAME,
                  tracking_uri=MLFLOW_URI)
with mlflow.start_run():
    # Create a simple model for demonstration
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, 3, activation="relu",
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Log model to registry
    model_info = mlflow.tensorflow.log_model(
        model, name="keras_model", registered_model_name="ImageClassifier"
    )

    # Tag for tracking
    mlflow.set_tags(
        {"model_type": "cnn", "dataset": "mnist", "framework": "keras"})

# Set alias for production deployment
client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name="ImageClassifier",
    alias="champion",
    version=model_info.registered_model_version,
)
