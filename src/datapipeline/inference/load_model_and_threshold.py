import mlflow
from mlflow.tracking import MlflowClient
import json
import os

def load_latest_model_and_threshold(registered_model_name: str,
                                    artifact_path_mlflow: str,
                                    file_name_threshold: str) :
    """
    Load the latest model and threshold from mlflow
    Args:
        registered_model_name (str): name of the model as registered in mlflow
        artifact_path_mlflow (str): path of the threshold artifact in mlflow
        file_name_threshold (str): name of the threshold file
    Returns
        model : catboost model object 
        threshold: clasiification threshold load from mlflow
    """

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{registered_model_name}'"  
    )

    if not versions:
        raise ValueError(f"No registered versions found for {registered_model_name}")

    latest = max(versions, key=lambda mv: int(mv.version))

    model = mlflow.catboost.load_model(
            f"models:/{registered_model_name}/{latest.version}"
    )

    threshold_path = mlflow.artifacts.download_artifacts(
        run_id=latest.run_id,
        artifact_path= os.path.join(artifact_path_mlflow, file_name_threshold)
    )

    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    return model, threshold
    