import pandas as pd
import argparse
from datapipeline.data.validate_schema import validate_schema
from datapipeline.data.hashing import compute_hash
from datapipeline.config.logging_config import setup_logging
import yaml   
from typing import Tuple
import mlflow
import logging

def load_raw_data(path:str, schema: dict, log_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load raw data from a CSV file, validate its schema, and compute a dataset hash.

    Args:
        path (str): The path to the CSV file.
        schema (dict): The schema of the dataset.

    Returns:
        tuple: A tuple containing the loaded DataFrame and the dataset hash.

    """
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Raw data loaded successfully")
    logger.info("Validating schema")
    validate_schema(df, schema)
    logger.info("Schema validated successfully")
    logger.info("Computing dataset hash")
    dataset_hash = compute_hash(path)
    logger.info("Dataset hash computed successfully")
    return df, dataset_hash



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")
    #parser.add_argument("--schema_path", type=str, required=True, help="Path to the schema file")
    #parser.add_argument("--experiment_name", type=str, help="Name of the experiment to be used in MLflow", 
    #                    default="Credit Card Fraud Detection")
    #parser.add_argument("--run_name", type=str, help="Name of the run to be used in MLflow", 
    #                    default="Data Ingestion")
    #parser.add_argument("--log_path", type=str, help="Path to the log file")

    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    with open(args.config_file_path, "r") as f:
            config = yaml.safe_load(f)

    mlflow.set_experiment(config['data_ingestion']["experiment_name"])
    
    with mlflow.start_run(run_name=config['data_ingestion']["run_name"]):
        mlflow.set_tag('stage', 'data_ingestion')
        schema_path = config['data_ingestion']["schema_path"]
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)

        mlflow.set_tag("schema_version", schema["schema_version"])
        mlflow.set_tag("dataset_name", schema["dataset_name"])

        log_path = config['data_ingestion']["log_path"]
        log_path = f"{log_path}/{mlflow.active_run().info.run_id}.log"

        dataset_path = config['data_ingestion']["dataset_path"]
        df, dataset_hash = load_raw_data(dataset_path, schema, log_path=log_path)

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_artifact(schema_path, artifact_path="schema")

        mlflow.log_metric("n_rows", df.shape[0])
        mlflow.log_metric("n_columns", df.shape[1])

        mlflow.log_artifact(__file__, artifact_path="code")