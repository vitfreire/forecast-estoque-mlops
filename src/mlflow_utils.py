import os
import mlflow


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "forecast_estoque_walmart")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("environment", os.getenv("APP_ENV", "prod"))
    mlflow.set_tag("project", "forecast-estoque-mlops")