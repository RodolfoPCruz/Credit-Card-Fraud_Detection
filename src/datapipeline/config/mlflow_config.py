import mlflow
import os
import logging



def setup_mlflow(experiment_name: str,
                logger: logging.Logger | None = None) -> None:
    """
    Configure MLflow tracking URI and experiment.

    Priority:
    1. MLFLOW_TRACKING_URI environment variable
    2. Local default (file-based backend)
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if logger:
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
