from mlflow.tracking.client import MlflowClient
import mlflow
import pandas as pd 
import logging
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

def get_latest_pipeline_run_id(
    experiment_name: str,
    run_name: str,
    pipeline_version: str,
    pipeline_status: str
) -> str:

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.runName = '{run_name}' "
            f"and tags.dataset_hash = '{pipeline_version}'"
            f"and tags.pipeline_status = '{pipeline_status}'"

        ),
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError(
            f"No pipeline_execution run found for git_commit='{pipeline_version}'"
        )
   
    return runs[0].info.run_id

def load_df_from_mlflow(experiment_name: str,
                        run_name: str,
                        pipeline_version: str,
                        artifact_path: str,
                        pipeline_status: str,
                        logger: logging.Logger | None = None) -> pd.DataFrame:
        
    run_id = get_latest_pipeline_run_id(experiment_name,
                                        run_name,
                                        pipeline_version,
                                        pipeline_status)


    if logger:
        logger.info(f"Loading artifact {artifact_path} from run {run_id}")

    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path
    )
    return pd.read_parquet(path)

