import argparse
import yaml
import mlflow
import logging
import pandas as pd

from enum import Enum
from datapipeline.training.load_data import load_raw_data
from datapipeline.training.hashing import compute_hash
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
from datapipeline.inference.load_model_and_threshold import load_latest_model_and_threshold
from datapipeline.inference.predict import predict
from datapipeline.pipeline.artifacts import get_git_commit
from datapipeline.pipeline.artifacts import load_df_from_mlflow

class Stage(str, Enum):
    INGEST = 'ingest'
    CLEAN = 'clean'
    SPLIT = 'split'
    CORRELATION = 'correlation'
    FEATURE_ENGINEERING = 'feature_engineering'
    MODEL_TRAINING = 'model_training'
    INFERENCE = 'inference' 

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

class PipelineRunner():

    def __init__(self, config_path: str, stage: Stage, run_correlation_diagnostics: bool):

        self.config_path = config_path
        self.stage = stage
        self.run_correlation_diagnostics = run_correlation_diagnostics  

        #Initial state of the pipeline
        self.df = None
        self.df_cleaned = None
        self.train_df = None
        self.test_df = None
        self.train_df_fe = None
        self.test_df_fe = None

        #The following will be initialized after
        self.config = None
        self.logger = None
        self.project_root = None
        self.schema = None
        self.git_commit = None

        self.execution_type = None


    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _bootstrap(self):

        self._load_config()

        self.project_root = get_project_root()

        #logging
        log_path = self.project_root / self.config['pipeline']['log_path']
        setup_logging(log_path)
        self.logger = logging.getLogger("pipeline")

        #schema
        self.schema_path = self.project_root / self.config['data_ingestion']["schema_path"]
        with open(self.schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

        #Dataset hash
        self.dataset_hash = compute_hash(
            self.project_root / self.config['data_ingestion']['raw_dataset_path'])

        #Git commit
        self.git_commit = get_git_commit()

        #MlFlow
        setup_mlflow(self.config['pipeline']['experiment_name'])

    def _get_stages_from(self, start_stage: str):
        if start_stage not in self.stage_order:
            raise ValueError(f"Invalid start stage: {start_stage}")
        
        start_index = self.stage_order.index(start_stage)
        return self.stage_order[start_index:]
    
    def _execute_pipeline(self, stages_to_run):
        for stage in stages_to_run:
            self.logger.info(f'Running stage: {stage}')
            self.stage_map[stage]()


    def run(self):

        self._bootstrap()

        
        self.stage_order = [
                Stage.INGEST,
                Stage.CLEAN,
                Stage.SPLIT,
                Stage.FEATURE_ENGINEERING,
                Stage.MODEL_TRAINING,
                Stage.INFERENCE
                      ]
        if (self.run_correlation_diagnostics or
            self.stage == Stage.CORRELATION):
            self.stage_order.insert(3,Stage.CORRELATION)

        self.stage_map = {
            Stage.INGEST: self._run_ingestion,
            Stage.CLEAN: self._run_cleaning,
            Stage.SPLIT: self._run_split,
            Stage.CORRELATION: self._run_correlation,
            Stage.FEATURE_ENGINEERING: self._run_feature_engineering,
            Stage.MODEL_TRAINING: self._run_training,
            Stage.INFERENCE: self._run_inference
        }

        self.run_name = self.config['pipeline']['run_name']

        if self.stage == Stage.INGEST:
            self.execution_type = 'full'
        else:
            self.execution_type = 'partial'
        self.logger.info(f'Starting pipeline from stage: {self.stage}')

        stages_to_run = self._get_stages_from(self.stage)


        with mlflow.start_run(run_name=self.run_name):

            self._set_global_tags()
          
            try:
                self._execute_pipeline(stages_to_run)
                mlflow.set_tag("pipeline_status", "completed")
                mlflow.set_tag("initial_stage", self.stage)

            except Exception:
                mlflow.set_tag("pipeline_status", "failed")
                raise
    

    def _set_global_tags(self):
        mlflow.set_tag("version", self.config['pipeline']['version'])
        mlflow.set_tag('schema_version', self.schema["schema_version"])
        mlflow.set_tag('dataset_name', self.schema["dataset_name"])
        mlflow.set_tag('git_commit', self.git_commit)
        mlflow.set_tag('dataset_hash', self.dataset_hash)
        mlflow.set_tag('execution_type', self.execution_type)

    def _run_ingestion(self):
        with mlflow.start_run(
            
            run_name=self.config['data_ingestion']['run_name'], 
                                    nested=True):
            self.df, self.dataset_hash = load_raw_data(
                dataset_path= self.project_root / self.config['data_ingestion']['raw_dataset_path'],
                schema=self.schema,
                logger=self.logger
                    )
            self.df.to_parquet(
                self.project_root / self.config['data_ingestion']['raw_dataset_path_parquet'])
            
            mlflow.log_param("raw_dataset_path", self.config['data_ingestion']['raw_dataset_path'])
            mlflow.log_metric("n_rows", self.df.shape[0])
            mlflow.log_metric("n_columns", self.df.shape[1])
        
        mlflow.log_param("dataset_hash", self.dataset_hash)
        mlflow.log_artifact(self.schema_path, artifact_path="schema")
        mlflow.log_artifact(
            self.project_root / self.config['data_ingestion']['raw_dataset_path_parquet'], 
            artifact_path=self.config['data_ingestion']['raw_dataset_path_mlflow'])
        
    def _run_cleaning(self):
        with mlflow.start_run(
            run_name=self.config['data_cleaning']['run_name'], nested=True):
            if self.df is None:
                self.df = load_df_from_mlflow(
                    experiment_name = self.config['pipeline']['experiment_name'],
                    run_name=self.run_name,
                    pipeline_version = self.dataset_hash,
                    artifact_path = self.config['data_ingestion']['raw_dataset_path_mlflow'],
                    pipeline_status = self.config['pipeline']['pipeline_status'],
                    logger=self.logger)
                
            self.df_cleaned, removed_rows = clean_data(
                df=self.df,
                target_column=self.config['data_cleaning']['target_column'],
                logger=self.logger
                )

            self.df_cleaned.to_parquet(
                self.project_root / self.config['data_cleaning']['cleaned_dataset_path'])

            validate_data(
                df=self.df_cleaned,
                target_column=self.config['data_cleaning']['target_column'],
                min_samples=self.config['data_cleaning']['min_samples'],
                num_classes=self.config['data_cleaning']['num_classes'],
                logger=self.logger
                ) 

            mlflow.log_metric("rows_removed", removed_rows)
        mlflow.log_artifact(
            self.project_root / self.config['data_cleaning']['cleaned_dataset_path'], 
            artifact_path=self.config['data_cleaning']['cleaned_dataset_path_mlflow'])
        
    def _run_split(self):
        with mlflow.start_run(
            run_name=self.config['data_split']['run_name'],
                         nested=True):
            test_size = self.config['data_split']['test_size']
            random_state = self.config['data_split']['random_state']
            target_column=  self.config['data_cleaning']['target_column']
                
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            if self.df_cleaned is None:
                self.df_cleaned = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['data_cleaning']['cleaned_dataset_path_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)

            self.train_df, self.test_df = split_data(
                df=self.df_cleaned,
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                logger=self.logger
                )

            validate_split(
                train_df=self.train_df,
                test_df=self.test_df,
                target_column=target_column,
                tolerance=self.config['data_split']['tolerance'],
                logger=self.logger
                )

            train_path = self.project_root / self.config['data_split']['train_path']
            test_path = self.project_root / self.config['data_split']['test_path']
            train_artifact_path_mlflow=self.config['data_split']['train_artifact_path_mlflow']
            test_artifact_path_mlflow=self.config['data_split']['test_artifact_path_mlflow']

            self.train_df.to_parquet(train_path)
            self.test_df.to_parquet(test_path)

        mlflow.log_artifact(train_path, artifact_path=train_artifact_path_mlflow)
        mlflow.log_artifact(test_path, artifact_path=test_artifact_path_mlflow)
    
    def _run_correlation(self):
        with mlflow.start_run(
            run_name=self.config['diagnostics']['run_name'], 
                nested=True):
            path_corr_matrix = self.project_root / self.config['diagnostics']['correlation_matrix']
            path_corr_metadata = self.project_root / self.config['diagnostics']['correlation_metadata']
            path_corr_heatmap =  self.project_root / self.config['diagnostics']['correlation_heatmap']
            correlation_path_mlflow = self.config['diagnostics']['correlation_path_mlflow']

            if self.train_df is None:
                self.train_df = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['data_split']['train_artifact_path_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)
                
            corr_matrix, corr_metadata = run_correlation_diagnostics(
                        self.train_df,
                        path_corr_heatmap,
                        logger=self.logger)

            corr_matrix.to_parquet(path_corr_matrix)
            corr_metadata.to_parquet(path_corr_metadata)
                
        mlflow.log_artifact(path_corr_matrix, artifact_path=correlation_path_mlflow)
        mlflow.log_artifact(path_corr_metadata, artifact_path=correlation_path_mlflow)
        mlflow.log_artifact(path_corr_heatmap, artifact_path=correlation_path_mlflow)

    def _run_feature_engineering(self):
        with mlflow.start_run(run_name=self.config['diagnostics']['run_name'], 
                                  nested=True):
            target_column = self.config['data_cleaning']['target_column']
            train_path_feature_engineered_mlflow = self.config['feature_engineering']['train_path_feature_engineered_mlflow']
            test_path_feature_engineered_mlflow = self.config['feature_engineering']['test_path_feature_engineered_mlflow']
            train_path_fe = self.project_root / self.config['feature_engineering']['train_path_feature_engineered']
            test_path_fe =  self.project_root / self.config['feature_engineering']['test_path_feature_engineered']
            remove_correlated_features = bool(self.config['feature_engineering']['remove_correlated_features'])
            threshold_correlation = self.config['feature_engineering']['threshold_correlation']
            self.logger.info(f'Removing correlated features: {remove_correlated_features}')

            if self.train_df is None: 
                self.train_df = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['data_split']['train_artifact_path_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)

            if self.test_df is None: 
                self.test_df = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['data_split']['test_artifact_path_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)
                
                
            self.logger.info(f'Correlation Threshold: {threshold_correlation}')
            mlflow.log_param("correlation_threshold", threshold_correlation)
            mlflow.log_param("remove correlated features", remove_correlated_features)
            features_to_remove = find_correlated_features(self.train_df, threshold_correlation)
            if features_to_remove:
                features_to_remove_path =  self.project_root / self.config['feature_engineering']['correlated_features_path']
                pd.DataFrame(features_to_remove, columns = 'feature').to_parquet(features_to_remove_path)
                mlflow.log_artifact(features_to_remove_path, 
                    artifact_path=self.config['feature_engineering']['correlated_features_path_mlflow'])
                
            self.train_df_fe , self.test_df_fe = apply_feature_engineering(
                    self.train_df, 
                    self.test_df, 
                    target_column, 
                    remove_correlated_features, 
                    threshold_correlation,
                    features_to_remove, 
                    logger=self.logger)
            self.train_df_fe.to_parquet(train_path_fe)
            self.test_df_fe.to_parquet(test_path_fe)
        mlflow.log_artifact(train_path_fe, artifact_path=train_path_feature_engineered_mlflow)
        mlflow.log_artifact(test_path_fe, artifact_path=test_path_feature_engineered_mlflow)

    def _run_training(self):

        with mlflow.start_run(run_name=self.config['model_training']['run_name'], 
                                  nested=True):
            target_column = self.config['data_cleaning']['target_column']
            #train_path_fe = self.project_root / self.config['feature_engineering']['train_path_feature_engineered']
            #test_path_fe = self.project_root / self.config['feature_engineering']['test_path_feature_engineered']

            if self.train_df_fe is None:
                self.train_df_fe = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['feature_engineering']['train_path_feature_engineered_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)

            if self.test_df_fe is None:
                self.test_df_fe = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['feature_engineering']['test_path_feature_engineered_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)

               
            classification_threshold_file_name = self.config['model_training']['classification_threshold_file_name']
            model_name = self.config['model_training']['model_name']

            hyperparameters = {}
            hyperparameters['random_state'] = self.config['model_training']['random_state']
            hyperparameters['l2_leaf_reg'] = self.config['model_training']['l2_leaf_reg']
            hyperparameters['depth'] = self.config['model_training']['depth']
            hyperparameters['iterations'] = self.config['model_training']['iterations']
            hyperparameters['learning_rate'] = self.config['model_training']['learning_rate']

            classification_threshold = float(self.config['model_training']['threshold'])
            artifact_path_model_training = self.config['model_training']['artifacts_path_model_training']
            registered_model_name = self.config['model_training']['registered_model_name']
            artifacts_path_mlflow_model_training = self.config['model_training']['artifacts_path_mlflow_model_training']

            results = train_model(
                train_df = self.train_df_fe,
                test_df= self.test_df_fe,
                target_column = target_column,
                hyperparameters = hyperparameters,
                threshold = classification_threshold,
                registered_model_name = registered_model_name,
                artifacts_dir=artifact_path_model_training,
                artifacts_path_mlflow = artifacts_path_mlflow_model_training,
                threshold_file_name = classification_threshold_file_name,
                logger = self.logger)

            mlflow.log_param('classification_threshold', results['Threshold'])
            mlflow.log_param('model_name', model_name)

            mlflow.log_metric('AUC', results['AUC'])
            mlflow.log_metric('Recall', results['Recall'])
            mlflow.log_metric('Precision', results['Precision'])
                
            mlflow.log_param('classification_threshold', results['Threshold'])
            mlflow.log_param('model_name', model_name)

    def _run_inference(self):
        with mlflow.start_run(run_name=self.config['inference']['run_name'], 
                    nested=True):
                
            artifacts_path_mlflow_model = self.config['model_training']['artifacts_path_mlflow_model_training']
            registered_model_name = self.config['model_training']['registered_model_name']

            model, threshold = load_latest_model_and_threshold(
                registered_model_name = registered_model_name,
                artifact_path_mlflow = artifacts_path_mlflow_model,
                file_name_threshold = self.config['model_training']['classification_threshold_file_name']
                )

            if self.test_df_fe is None:
                self.test_df_fe = load_df_from_mlflow(
                experiment_name = self.config['pipeline']['experiment_name'],
                run_name = self.run_name,
                pipeline_version = self.dataset_hash,
                artifact_path = self.config['feature_engineering']['test_path_feature_engineered_mlflow'],
                pipeline_status = self.config['pipeline']['pipeline_status'],
                logger=self.logger)
               
            target_column = self.config['data_cleaning']['target_column']
            predictions_path = self.config['inference']['artifacts_path_inference']
            artifacts_path_predicitions_mlflow = self.config['inference']['artifacts_path_inference_mlflow']
            self.logger.info('Generating Predictions')
            predictions = predict(model = model,
                                    data = self.test_df_fe,
                                    threshold = threshold,
                                    target_column = target_column)
            predictions.to_parquet(predictions_path)
        mlflow.log_artifact(predictions_path, artifact_path=artifacts_path_predicitions_mlflow)
                           
    

def main():

    args = parse_args()

    runner = PipelineRunner(
        config_path=args.config,
        stage=args.stage,
        run_correlation_diagnostics=args.run_correlation_diagnostics
    )

    runner.run()
    
    
    










def main_2():


    args = parse_args()

    df = None
    df_cleaned = None
    train_df = None
    test_df = None
    train_df_fe = None
    test_df_fe = None
    
    PIPELINE_STAGES = [
                Stage.INGEST,
                Stage.CLEAN,
                Stage.SPLIT,
                Stage.FEATURE_ENGINEERING,
                Stage.MODEL_TRAINING,
                Stage.INFERENCE
                      ]
    if (args.run_correlation_diagnostics or
        args.stage == Stage.CORRELATION):
        PIPELINE_STAGES.insert(3,Stage.CORRELATION)


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    PROJECT_ROOT = get_project_root()

    # Logging
    log_path = PROJECT_ROOT / config['pipeline']['log_path']
    setup_logging(log_path)
    logger = logging.getLogger('pipeline')

    schema_path = PROJECT_ROOT / config['data_ingestion']["schema_path"]
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)
    schema_version = schema["schema_version"]
    dataset_name = schema["dataset_name"]
    dataset_hash = compute_hash(PROJECT_ROOT / config['data_ingestion']['raw_dataset_path'])
    git_commit = get_git_commit()

    #MLFlow
    mlflow_experiment_name = config['pipeline']['experiment_name']
    setup_mlflow(mlflow_experiment_name)
    run_name = config['pipeline']['run_name']
    run_tag = config['pipeline']['run_tag']
    pipeline_status = config['pipeline']['pipeline_status']

    with mlflow.start_run(run_name = run_name):

        mlflow.set_tag("version", run_tag)
        mlflow.set_tag('schema_version', schema_version)
        mlflow.set_tag('dataset_name', dataset_name)
        mlflow.set_tag('git_commit', git_commit)
        mlflow.set_tag('dataset_hash', dataset_hash)

        start_idx = PIPELINE_STAGES.index(args.stage)
        logger.info(f'Starting pipeline from stage: {args.stage}')

        #---------------Data Ingestion----------------------------------------
        if Stage.INGEST in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['data_ingestion']['run_name'], 
                                    nested=True):
                df, dataset_hash = load_raw_data(
                    dataset_path= PROJECT_ROOT / config['data_ingestion']['raw_dataset_path'],
                    schema=schema,
                    logger=logger
                )
                df.to_parquet(PROJECT_ROOT / config['data_ingestion']['raw_dataset_path_parquet'])
                mlflow.log_param("raw_dataset_path", config['data_ingestion']['raw_dataset_path'])

                mlflow.log_metric("n_rows", df.shape[0])
                mlflow.log_metric("n_columns", df.shape[1])

            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_artifact(schema_path, artifact_path="schema")
            mlflow.log_artifact(PROJECT_ROOT / config['data_ingestion']['raw_dataset_path_parquet'], 
                                    artifact_path=config['data_ingestion']['raw_dataset_path_mlflow'])

        #---------------Data Cleaning-----------------------------------------
        if Stage.CLEAN in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['data_cleaning']['run_name'], nested=True):

                if df is None:
                    df = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name=run_name,
                        pipeline_version =dataset_hash,
                        artifact_path = config['data_ingestion']['raw_dataset_path_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)
                
                df_cleaned, removed_rows = clean_data(
                    df=df,
                    target_column=config['data_cleaning']['target_column'],
                    logger=logger
                )

                df_cleaned.to_parquet(PROJECT_ROOT / config['data_cleaning']['cleaned_dataset_path'])

                validate_data(
                    df=df_cleaned,
                    target_column=config['data_cleaning']['target_column'],
                    min_samples=config['data_cleaning']['min_samples'],
                    num_classes=config['data_cleaning']['num_classes'],
                    logger=logger
                ) 

                mlflow.log_metric("rows_removed", removed_rows)
            mlflow.log_artifact(PROJECT_ROOT / config['data_cleaning']['cleaned_dataset_path'], 
                                    artifact_path=config['data_cleaning']['cleaned_dataset_path_mlflow'])


        #---------------Data Split--------------------------------------------
        if Stage.SPLIT in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['data_split']['run_name'],
                                 nested=True):
                test_size = config['data_split']['test_size']
                random_state = config['data_split']['random_state']
                target_column=config['data_cleaning']['target_column']
                
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)

                if df_cleaned is None:
                    df_cleaned = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['data_cleaning']['cleaned_dataset_path_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)


                train_df, test_df = split_data(
                    df=df_cleaned,
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

        #---------------Correlation-Diagnostics-------------------------------
        if Stage.CORRELATION in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['diagnostics']['run_name'], 
                                  nested=True):
                path_corr_matrix = PROJECT_ROOT / config['diagnostics']['correlation_matrix']
                path_corr_metadata = PROJECT_ROOT / config['diagnostics']['correlation_metadata']
                path_corr_heatmap =  PROJECT_ROOT / config['diagnostics']['correlation_heatmap']
                correlation_path_mlflow = config['diagnostics']['correlation_path_mlflow']

                if train_df is None:
                    train_df = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['data_split']['train_artifact_path_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)
                
                corr_matrix, corr_metadata = run_correlation_diagnostics(train_df,
                    path_corr_heatmap,
                    logger=logger)

                corr_matrix.to_parquet(path_corr_matrix)
                corr_metadata.to_parquet(path_corr_metadata)
                
            mlflow.log_artifact(path_corr_matrix, artifact_path=correlation_path_mlflow)
            mlflow.log_artifact(path_corr_metadata, artifact_path=correlation_path_mlflow)
            mlflow.log_artifact(path_corr_heatmap, artifact_path=correlation_path_mlflow)

        #---------------Feature Engineeruing----------------------------------
        if Stage.FEATURE_ENGINEERING in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['diagnostics']['run_name'], 
                                  nested=True):
                target_column = config['data_cleaning']['target_column']
                train_path_feature_engineered_mlflow = config['feature_engineering']['train_path_feature_engineered_mlflow']
                test_path_feature_engineered_mlflow = config['feature_engineering']['test_path_feature_engineered_mlflow']
                train_path_fe = PROJECT_ROOT / config['feature_engineering']['train_path_feature_engineered']
                test_path_fe = PROJECT_ROOT / config['feature_engineering']['test_path_feature_engineered']
                remove_correlated_features = bool(config['feature_engineering']['remove_correlated_features'])
                threshold_correlation = config['feature_engineering']['threshold_correlation']
                logger.info(f'Removing correlated features: {remove_correlated_features}')

                if train_df is None: 
                    train_df = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['data_split']['train_artifact_path_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)

                if test_df is None: 
                    test_df = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['data_split']['test_artifact_path_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)
                
                
                logger.info(f'Correlation Threshold: {threshold_correlation}')
                mlflow.log_param("correlation_threshold", threshold_correlation)
                mlflow.log_param("remove correlated features", remove_correlated_features)
                features_to_remove = find_correlated_features(train_df, threshold_correlation)
                if features_to_remove:
                    features_to_remove_path =  PROJECT_ROOT / config['feature_engineering']['correlated_features_path']
                    pd.DataFrame(features_to_remove, columns = 'feature').to_parquet(features_to_remove_path)
                    mlflow.log_artifact(features_to_remove_path, 
                                        artifact_path=config['feature_engineering']['correlated_features_path_mlflow'])
                train_df_fe , test_df_fe = apply_feature_engineering(train_df, 
                                                                    test_df, 
                                                                    target_column, 
                                                                    remove_correlated_features, 
                                                                    threshold_correlation,
                                                                    features_to_remove, 
                                                                    logger=logger)
                train_df_fe.to_parquet(train_path_fe)
                test_df_fe.to_parquet(test_path_fe)
            mlflow.log_artifact(train_path_fe, artifact_path=train_path_feature_engineered_mlflow)
            mlflow.log_artifact(test_path_fe, artifact_path=test_path_feature_engineered_mlflow)

        #---------------Model Training----------------------------------------
        if Stage.MODEL_TRAINING in PIPELINE_STAGES[start_idx:]:
            with mlflow.start_run(run_name=config['model_training']['run_name'], 
                                  nested=True):
                target_column = config['data_cleaning']['target_column']
                train_path_fe = PROJECT_ROOT / config['feature_engineering']['train_path_feature_engineered']
                test_path_fe = PROJECT_ROOT / config['feature_engineering']['test_path_feature_engineered']

                if train_df_fe is None:
                    train_df_fe = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['feature_engineering']['train_path_feature_engineered_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)

                if test_df_fe is None:
                    test_df_fe = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['feature_engineering']['test_path_feature_engineered_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)

               
                classification_threshold_file_name = config['model_training']['classification_threshold_file_name']
                model_name = config['model_training']['model_name']

                hyperparameters = {}
                hyperparameters['random_state'] = config['model_training']['random_state']
                hyperparameters['l2_leaf_reg'] = config['model_training']['l2_leaf_reg']
                hyperparameters['depth'] = config['model_training']['depth']
                hyperparameters['iterations'] = config['model_training']['iterations']
                hyperparameters['learning_rate'] = config['model_training']['learning_rate']

                classification_threshold = float(config['model_training']['threshold'])
                artifact_path_model_training = config['model_training']['artifacts_path_model_training']
                registered_model_name = config['model_training']['registered_model_name']
                artifacts_path_mlflow_model_training = config['model_training']['artifacts_path_mlflow_model_training']

                results = train_model(
                    train_df = train_df_fe,
                    test_df= test_df_fe,
                    target_column = target_column,
                    hyperparameters = hyperparameters,
                    threshold = classification_threshold,
                    registered_model_name = registered_model_name,
                    artifacts_dir=artifact_path_model_training,
                    artifacts_path_mlflow = artifacts_path_mlflow_model_training,
                    threshold_file_name = classification_threshold_file_name,
                    logger = logger)

                mlflow.log_param('classification_threshold', results['Threshold'])
                mlflow.log_param('model_name', model_name)

                mlflow.log_metric('AUC', results['AUC'])
                mlflow.log_metric('Recall', results['Recall'])
                mlflow.log_metric('Precision', results['Precision'])
            
            mlflow.log_param('classification_threshold', results['Threshold'])
            mlflow.log_param('model_name', model_name)

        #---------------Inference---------------------------------------------
        if Stage.INFERENCE in PIPELINE_STAGES[start_idx:]:
             with mlflow.start_run(run_name=config['inference']['run_name'], 
                                  nested=True):
                
                artifacts_path_mlflow_model = config['model_training']['artifacts_path_mlflow_model_training']
                registered_model_name = config['model_training']['registered_model_name']

                model, threshold = load_latest_model_and_threshold(
                    registered_model_name = registered_model_name,
                    artifact_path_mlflow = artifacts_path_mlflow_model,
                    file_name_threshold = config['model_training']['classification_threshold_file_name']
                )

                if test_df_fe is None:
                    test_df_fe = load_df_from_mlflow(
                        experiment_name = config['pipeline']['experiment_name'],
                        run_name = run_name,
                        pipeline_version = dataset_hash,
                        artifact_path = config['feature_engineering']['test_path_feature_engineered_mlflow'],
                        pipeline_status = pipeline_status,
                        logger=logger)
               
                target_column = config['data_cleaning']['target_column']

                predictions_path = config['inference']['artifacts_path_inference']
                artifacts_path_predicitions_mlflow = config['inference']['artifacts_path_inference_mlflow']

                predictions = predict(model = model,
                                    data = test_df_fe,
                                    threshold = threshold,
                                    target_column = target_column)

                predictions.to_parquet(predictions_path)
                mlflow.log_artifact(predictions_path, artifact_path=artifacts_path_predicitions_mlflow)
                           
        try:
        # pipeline inteira
            mlflow.set_tag("pipeline_status", "completed")
            mlflow.set_tag("initial_stage", args.stage)
        except Exception:
            mlflow.set_tag("pipeline_status", "failed")
            raise


if __name__ == "__main__":
    main()