import argparse
import yaml
import mlflow
import logging

from enum import Enum
from datapipeline.data.load_data import load_raw_data
from datapipeline.data.clean_data import clean_data
from datapipeline.data.split_data import split_data
from datapipeline.data.validate_data import validate_data
from datapipeline.data.validate_split import validate_split
from datapipeline.config.logging_config import setup_logging

class Stage(str, Enum):
    INGEST = 'ingest'
    CLEAN = 'clean'
    SPLIT = 'split'

PIPELINE_STAGES = [
    Stage.INGEST,
    Stage.CLEAN,
    Stage.SPLIT
]

def parse_args():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the configuration file',
        required=True
    )
    parser.add_argument(
        '--stage',
        type=Stage,
        help='Stage of the pipeline',
        default=Stage.INGEST, 
        choices=list(Stage)
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Logging
    log_path = config['pipeline']['log_path']
    setup_logging(log_path)
    logger = logging.getLogger('pipeline')

    #MLFlow
    mlflow.set_experiment(config['pipeline']['experiment_name'])

    with mlflow.start_run(run_name='pipeline_execution'):

        start_idx = PIPELINE_STAGES.index(args.stage)
        logger.info(f'Starting pipeline from stage: {args.stage}')

        #---------------Data Ingestion----------------------------
        if Stage.INGEST in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='data_ingestion', nested=True):
                schema_path = config['data_ingestion']["schema_path"]
                with open(schema_path, "r") as f:
                    schema = yaml.safe_load(f)
                df, dataset_hash = load_raw_data(
                    dataset_path=config['data_ingestion']['dataset_path'],
                    schema=schema,
                    logger=logger
                )

                mlflow.set_tag("schema_version", schema["schema_version"])
                mlflow.set_tag("dataset_name", schema["dataset_name"])

                mlflow.log_param("dataset_hash", dataset_hash)
                mlflow.log_param("dataset_path", config['data_ingestion']['dataset_path'])

                mlflow.log_metric("n_rows", df.shape[0])
                mlflow.log_metric("n_columns", df.shape[1])

                mlflow.log_artifact(schema_path, artifact_path="schema")

        #---------------Data Cleaning----------------------------
        if Stage.CLEAN in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='data_cleaning', nested=True):
                df, removed_rows = clean_data(
                    df=df,
                    target_column=config['data']['target_column'],
                    logger=logger
                )

                validate_data(
                    df=df,
                    target_column=config['data']['target_column'],
                    min_samples=config['data']['min_samples'],
                    num_classes=config['data']['num_classes'],
                    logger=logger
                ) 

                mlflow.log_metric("rows_removed", removed_rows)

        #---------------Data Split--------------------------------
        if Stage.SPLIT in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='data_split', nested=True):
                test_size = config['data_split']['test_size']
                random_state = config['data_split']['random_state']
                target_column=config['data']['target_column']
                
                mlflow.log_metric("test_size", test_size)
                mlflow.log_param("random_state", random_state)

                train_df, test_df = split_data(
                    df=df,
                    target_column=target_column,
                    test_size=test_size,
                    random_state=random_state,
                    logger=logger
                )

                validate_split(
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    tolerance=config['data']['tolerance'],
                    logger=logger
                )

                train_path = config['data_split']['train_path']
                test_path = config['data_split']['test_path']

                train_df.to_parquet(train_path)
                test_df.to_parquet(test_path)

                mlflow.log_artifact(train_path, artifact_path="train")
                mlflow.log_artifact(test_path, artifact_path="test")


if __name__ == "__main__":
    main()