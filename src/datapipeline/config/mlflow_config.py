import mlflow
import os
import logging
from pathlib import Path


def setup_mlflow(experiment_name: str,
                local_folder_path: str,
                logger: logging.Logger | None = None) -> None:
    """
    Configure MLflow tracking URI and experiment.

    Priority:
    1. MLFLOW_TRACKING_URI environment variable
    2. Local default (file-based backend)
    """

    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if env_tracking_uri:
        tracking_uri = env_tracking_uri
    else:
        path = Path(local_folder_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{path}"
    print(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if logger:
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
