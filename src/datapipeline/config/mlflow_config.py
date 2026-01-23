from pathlib import Path
import mlflow


def get_project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except NameError:
        # Notebook
        return Path.cwd().resolve().parents[1]

PROJECT_ROOT = get_project_root()

MLFLOW_DIR = PROJECT_ROOT / "mlflow" / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = (PROJECT_ROOT /'mlflow'/"mlruns").as_uri()

def setup_mlflow(experiment_name: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)