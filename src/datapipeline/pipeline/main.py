import argparse
import yaml
import mlflow
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from enum import Enum
from datapipeline.training.load_data import load_raw_data
from datapipeline.training.clean_data import clean_data
from datapipeline.training.split_data import split_data
from datapipeline.training.validate_data import validate_data
from datapipeline.training.validate_split import validate_split
from datapipeline.training.model_training import train_model
from datapipeline.diagnostics.run_correlation_diagnostics import run_correlation_diagnostics
from datapipeline.features.correlation_pipeline import find_correlated_features
from datapipeline.features.feature_engineering import apply_feature_engineering
from datapipeline.config.logging_config import setup_logging
from datapipeline.config.mlflow_config import setup_mlflow
from datapipeline.config.mlflow_config import get_project_root


class Stage(str, Enum):
    INGEST = 'ingest'
    CLEAN = 'clean'
    SPLIT = 'split'
    CORRELATION = 'correlation'
    FEATURE_ENGINEERING = 'feature_engineering'
    MODEL_TRAINING = 'model_training'


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
    parser.add_argument(
        '--run_correlation_diagnostics',
        action='store_true',
        help='whether to run correlation diagnostics')
    
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    PIPELINE_STAGES = [
                Stage.INGEST,
                Stage.CLEAN,
                Stage.SPLIT,
                Stage.FEATURE_ENGINEERING,
                Stage.MODEL_TRAINING
                      ]
    if args.run_correlation_diagnostics:
        PIPELINE_STAGES.insert(3,Stage.CORRELATION)


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    PROJECT_ROOT = get_project_root()

    # Logging
    log_path = PROJECT_ROOT / config['pipeline']['log_path']
    setup_logging(log_path)
    logger = logging.getLogger('pipeline')

    #MLFlow
    mlflow_experiment_name = config['pipeline']['experiment_name']
    mlflow.set_experiment(mlflow_experiment_name)
    setup_mlflow(mlflow_experiment_name)

    with mlflow.start_run(run_name='pipeline_execution'):

        start_idx = PIPELINE_STAGES.index(args.stage)
        logger.info(f'Starting pipeline from stage: {args.stage}')

        #---------------Data Ingestion----------------------------
        if Stage.INGEST in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='data_ingestion', nested=True):
                schema_path = PROJECT_ROOT / config['data_ingestion']["schema_path"]
                with open(schema_path, "r") as f:
                    schema = yaml.safe_load(f)
                df, dataset_hash = load_raw_data(
                    dataset_path= PROJECT_ROOT / config['data_ingestion']['dataset_path'],
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
                    target_column=config['data_cleaning']['target_column'],
                    logger=logger
                )

                validate_data(
                    df=df,
                    target_column=config['data_cleaning']['target_column'],
                    min_samples=config['data_cleaning']['min_samples'],
                    num_classes=config['data_cleaning']['num_classes'],
                    logger=logger
                ) 

                mlflow.log_metric("rows_removed", removed_rows)

        #---------------Data Split--------------------------------
        if Stage.SPLIT in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='data_split', nested=True):
                test_size = config['data_split']['test_size']
                random_state = config['data_split']['random_state']
                target_column=config['data_cleaning']['target_column']
                
                mlflow.log_param("test_size", test_size)
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
                    tolerance=config['data_split']['tolerance'],
                    logger=logger
                )

                train_path = PROJECT_ROOT / config['data_split']['train_path']
                test_path = PROJECT_ROOT / config['data_split']['test_path']
                train_artifact_path_mlflow=config['data_split']['train_artifact_path_mlflow']
                test_artifact_path_mlflow=config['data_split']['test_artifact_path_mlflow']

                train_df.to_parquet(train_path)
                test_df.to_parquet(test_path)

                mlflow.log_artifact(train_path, artifact_path=train_artifact_path_mlflow)
                mlflow.log_artifact(test_path, artifact_path=test_artifact_path_mlflow)

        #---------------Correlation-Diagnostics--------------------------------
        if Stage.CORRELATION in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='correlation diagnostics', 
                                  nested=True):
                path_corr_matrix = PROJECT_ROOT / config['diagnostics']['correlation_matrix']
                path_corr_metadata = PROJECT_ROOT / config['diagnostics']['correlation_metadata']
                path_corr_heatmap =  PROJECT_ROOT / config['diagnostics']['correlation_heatmap']
                correlation_path_mlflow = config['diagnostics']['correlation_path_mlflow']
                
                corr_matrix, corr_metadata = run_correlation_diagnostics(train_df,
                    logger=logger)

                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix)

                plt.savefig(path_corr_heatmap, dpi = 400,
                            bbox_inches='tight')

                corr_matrix.to_parquet(path_corr_matrix)
                corr_metadata.to_parquet(path_corr_metadata)
                
                mlflow.log_artifact(path_corr_matrix, artifact_path=correlation_path_mlflow)
                mlflow.log_artifact(path_corr_metadata, artifact_path=correlation_path_mlflow)
                mlflow.log_artifact(path_corr_heatmap, artifact_path=correlation_path_mlflow)

        #---------------Feature Engineeruing--------------------------------
        if Stage.FEATURE_ENGINEERING in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='feature engineering', 
                                  nested=True):
                target_column = config['data_cleaning']['target_column']
                fe_path_mlflow = config['feature_engineering']['fe_path_mlflow']
                train_path_fe = PROJECT_ROOT / config['feature_engineering']['train_path_feature_engineered']
                test_path_fe = PROJECT_ROOT / config['feature_engineering']['test_path_feature_engineered']
                remove_correlated_features = bool(config['feature_engineering']['remove_correlated_features'])
                threshold = config['feature_engineering']['threshold']
                logger.info(f'Removing correlated features: {remove_correlated_features}')

                if remove_correlated_features:
                    logger.info(f'Correlation Threshold: {threshold}')
                    mlflow.log_param("correlation_threshold", threshold)
                    mlflow.log_param("remove correlated features", remove_correlated_features)
                    features_to_remove = find_correlated_features(train_df, threshold)
                    if features_to_remove:
                        features_to_remove_path =  PROJECT_ROOT / config['feature_engineering']['correlated_features']
                        pd.DataFrame(features_to_remove, columns = 'feature').to_parquet(features_to_remove_path)
                        mlflow.log_artifact(features_to_remove_path, artifact_path=fe_path_mlflow)
                    train_df_fe , test_df_fe = apply_feature_engineering(train_df, 
                                                                         test_df, 
                                                                         target_column, 
                                                                         remove_correlated_features, 
                                                                         threshold,
                                                                         features_to_remove, 
                                                                         logger=logger)
                    train_df_fe.to_parquet(train_path_fe)
                    test_df_fe.to_parquet(test_path_fe)
                    mlflow.log_artifact(train_path_fe, artifact_path=fe_path_mlflow)
                    mlflow.log_artifact(test_path_fe, artifact_path=fe_path_mlflow)

        #---------------Model Training----------------------------
        if Stage.MODEL_TRAINING in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name='Model_training', 
                                  nested=True):
                target_column = config['data_cleaning']['target_column']
                train_path_fe = PROJECT_ROOT / config['feature_engineering']['train_path_feature_engineered']
                test_path_fe = PROJECT_ROOT / config['feature_engineering']['test_path_feature_engineered']
                train_df_fe = pd.read_parquet(train_path_fe)
                test_df_fe = pd.read_parquet(test_path_fe)
                threshold_file_name = config['model_training']['threshold_file_name']

                model_name = config['model_training']['model_name']

                hyperparameters = {}
                hyperparameters['random_state'] = config['model_training']['random_state']
                hyperparameters['l2_leaf_reg'] = config['model_training']['l2_leaf_reg']
                hyperparameters['depth'] = config['model_training']['depth']
                hyperparameters['iterations'] = config['model_training']['iterations']
                hyperparameters['learning_rate'] = config['model_training']['learning_rate']

                classification_threshold = float(config['model_training']['threshold'])
                artifact_path = config['model_training']['artifacts_path']
                registered_model_name = config['model_training']['registered_model_name']
                artifacts_path_mlflow = config['model_training']['artifacts_path_mlflow']


                results = train_model(
                    train_df = train_df_fe,
                    test_df= test_df_fe,
                    target_column = target_column,
                    hyperparameters = hyperparameters,
                    threshold = classification_threshold,
                    registered_model_name = registered_model_name,
                    artifacts_dir=artifact_path,
                    artifacts_path_mlflow = artifacts_path_mlflow,
                    threshold_file_name = threshold_file_name,
                    logger = logger)

                mlflow.log_param('classification_thresold', results['Threshold'])
                mlflow.log_param('model_name', model_name)

                mlflow.log_metric('AUC', results['AUC'])
                mlflow.log_metric('Recall', results['Recall'])
                mlflow.log_metric('Precision', results['Precision'])

if __name__ == "__main__":
    main()