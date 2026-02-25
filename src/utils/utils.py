import os

import mlflow


def configurar_mlflow(
        tracking_uri: str, experiment_name: str = "regressao_fipe_v2",

):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri