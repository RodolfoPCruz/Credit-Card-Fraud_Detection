import pandas as pd
import argparse
from datapipeline.training.validate_schema import validate_schema
from datapipeline.training.hashing import compute_hash
from typing import Tuple
import logging

def load_raw_data(dataset_path, 
                  schema: dict, 
                  logger: logging.Logger | None = None
) -> Tuple[pd.DataFrame, str]:
    """
    Load raw data from a CSV file, validate its schema, and compute a dataset hash.

    Args:
        dataset_path (str): The path to the CSV file.
        schema (dict): The schema of the dataset.
        logger (logging.Logger, optional): Logger instance.
    Returns:
        tuple: A tuple containing the loaded DataFrame and the dataset hash.

    """
    if logger:
        logger.info("Loading raw data from %s", dataset_path)
    
    df = pd.read_csv(dataset_path)
    
    if logger:
        logger.info("Validating schema")
    validate_schema(df, schema)

    if logger:
        logger.info("Computing dataset hash")
    dataset_hash = compute_hash(dataset_path)

    return df, dataset_hash


'''
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    with open(args.config_file_path, "r") as f:
            config = yaml.safe_load(f)

    mlflow.set_experiment(config['data_ingestion']["experiment_name"])
    
    run_name = config['data_ingestion']["run_name"]
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('stage', 'data_ingestion')
        schema_path = config['data_ingestion']["schema_path"]
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)

        mlflow.set_tag("schema_version", schema["schema_version"])
        mlflow.set_tag("dataset_name", schema["dataset_name"])

        log_path = config['data_ingestion']["log_path"]
        log_path = f"{log_path}/{run_name}_{mlflow.active_run().info.run_id}.log"

        dataset_path = config['data_ingestion']["dataset_path"]
        df, dataset_hash = load_raw_data(dataset_path, schema, log_path=log_path)

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_artifact(schema_path, artifact_path="schema")

        mlflow.log_metric("n_rows", df.shape[0])
        mlflow.log_metric("n_columns", df.shape[1])

        mlflow.log_artifact(__file__, artifact_path="code")
'''