import argparse
import yaml
import mlflow
import logging
import pandas as pd
from pathlib import Path

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
from datapipeline.inference.load_model_and_threshold import load_latest_model_and_threshold
from datapipeline.inference.predict import predict
from datapipeline.pipeline.artifacts import get_git_commit
from datapipeline.pipeline.artifacts import load_df_from_mlflow
from datapipeline.pipeline.artifacts import log_dataframe_artifact
from datapipeline.pipeline.artifacts import get_log_dir


class Stage(str, Enum):
    INGEST = 'ingest'
    CLEAN = 'clean'
    SPLIT = 'split'
    CORRELATION = 'correlation'
    FEATURE_ENGINEERING = 'feature_engineering'
    MODEL_TRAINING = 'model_training'
    EVALUATION = 'evaluation' 

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

        self.config_path = Path(config_path).resolve()
        self.project_root = Path(self.config_path.parent)
        self.stage = stage
        self.run_correlation_diagnostics = run_correlation_diagnostics  

        #The following will be initialized after
        self.config = None
        self.logger = None
        self.schema = None
        self.git_commit = None

        self.execution_type = None


    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _bootstrap(self):

        self._load_config()


        #logging
        log_path = get_log_dir()
        log_path =  log_path / self.config['pipeline']['log_path']
        setup_logging(log_path)
        self.logger = logging.getLogger("pipeline")

        #schema
        self.schema_path = self.project_root / self.config[Stage.INGEST]["schema_path"]
        with open(self.schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

        #Dataset hash
        self.dataset_hash = compute_hash(
            self.project_root / self.config[Stage.INGEST]['input_paths'][0])

        #Git commit
        self.git_commit = get_git_commit()

        #MlFlow
        setup_mlflow(self.config['pipeline']['experiment_name'], self.logger)

    def _get_stages_from(self, start_stage: str):
        if start_stage not in self.stage_order:
            raise ValueError(f"Invalid start stage: {start_stage}")
        
        start_index = self.stage_order.index(start_stage)
        return self.stage_order[start_index:]
    
    def _load_input(self, stage):
        input_paths = self.config[stage]['input_paths']
        inputs = []
        for path in input_paths:
            full_path = self.project_root / path
            print(full_path)
            if not full_path.exists():
                raise FileNotFoundError(f"Input file not found for stage {stage}: {full_path}")
            inputs.append(pd.read_parquet(full_path))
        return inputs
    
    def _execute_pipeline(self, stages_to_run):
        for stage in stages_to_run:
            self.logger.info(f'Running stage: {stage}')
            if stage == Stage.INGEST:
                self.stage_map[stage]()
            #elif stage == Stage.CLEAN:
            #    inputs = self._load_input(stage)
            #    self.stage_map[stage](*inputs)
            else:
                inputs = self._load_input(stage)
                self.stage_map[stage](*inputs)


    def run(self):

        self._bootstrap()

        self.stage_order = [
                Stage.INGEST,
                Stage.CLEAN,
                Stage.SPLIT,
                Stage.FEATURE_ENGINEERING,
                Stage.MODEL_TRAINING,
                Stage.EVALUATION
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
            Stage.EVALUATION: self._model_evaluation
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
            
            run_name=self.config[Stage.INGEST]['run_name'], 
                                    nested=True):
            df, dataset_hash = load_raw_data(
                dataset_path= self.project_root / self.config[Stage.INGEST]['input_paths'][0],
                schema=self.schema,
                logger=self.logger
                    )
            
            output_dir = self.project_root / self.config[Stage.INGEST]['raw_dataset_path_parquet']
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(
                output_dir)
            
            mlflow.log_param("raw_dataset_path", self.config[Stage.INGEST]['input_paths'][0])
            mlflow.log_metric("n_rows", df.shape[0])
            mlflow.log_metric("n_columns", df.shape[1])
        
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_artifact(self.schema_path, artifact_path="schema")
        mlflow.log_artifact(
            output_dir, 
            artifact_path=self.config[Stage.INGEST]['raw_dataset_path_mlflow'])
        
    def _run_cleaning(self, raw_data: pd.DataFrame):
        with mlflow.start_run(
            run_name=self.config[Stage.CLEAN]['run_name'], nested=True):
            
            df_cleaned, removed_rows = clean_data(
                df=raw_data,
                target_column=self.config[Stage.CLEAN]['target_column'],
                logger=self.logger
                )

            output_dir = self.project_root / self.config[Stage.CLEAN]['cleaned_dataset_path']
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            df_cleaned.to_parquet(
                output_dir)

            validate_data(
                df=df_cleaned,
                target_column=self.config[Stage.CLEAN]['target_column'],
                min_samples=self.config[Stage.CLEAN]['min_samples'],
                num_classes=self.config[Stage.CLEAN]['num_classes'],
                logger=self.logger
                ) 

            mlflow.log_metric("rows_removed", removed_rows)
        mlflow.log_artifact(
            output_dir, 
            artifact_path=self.config[Stage.CLEAN]['cleaned_dataset_path_mlflow'])
        
    def _run_split(self, cleaned_data: pd.DataFrame):
        with mlflow.start_run(
            run_name=self.config[Stage.SPLIT]['run_name'],
                         nested=True):
            test_size = self.config[Stage.SPLIT]['test_size']
            random_state = self.config[Stage.SPLIT]['random_state']
            target_column=  self.config[Stage.CLEAN]['target_column']
                
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
                
            train_df, test_df = split_data(
                df=cleaned_data,
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                logger=self.logger
                )

            validate_split(
                train_df=train_df,
                test_df=test_df,
                target_column=target_column,
                tolerance=self.config[Stage.SPLIT]['tolerance'],
                logger=self.logger
                )

            train_path = self.project_root / self.config[Stage.SPLIT]['train_path']
            test_path = self.project_root / self.config[Stage.SPLIT]['test_path']
            train_artifact_path_mlflow=self.config[Stage.SPLIT]['train_artifact_path_mlflow']
            test_artifact_path_mlflow=self.config[Stage.SPLIT]['test_artifact_path_mlflow']

            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_parquet(train_path)
            test_df.to_parquet(test_path)

        mlflow.log_artifact(train_path, artifact_path=train_artifact_path_mlflow)
        mlflow.log_artifact(test_path, artifact_path=test_artifact_path_mlflow)
    
    def _run_correlation(self, df: pd.DataFrame):
        with mlflow.start_run(
            run_name=self.config[Stage.CORRELATION]['run_name'], 
                nested=True):
            path_corr_matrix = self.project_root / self.config[Stage.CORRELATION]['correlation_matrix']
            path_corr_metadata = self.project_root / self.config[Stage.CORRELATION]['correlation_metadata']
            path_corr_heatmap =  self.project_root / self.config[Stage.CORRELATION]['correlation_heatmap']
            correlation_path_mlflow = self.config[Stage.CORRELATION]['correlation_path_mlflow']


            corr_matrix, corr_metadata = run_correlation_diagnostics(
                        df,
                        path_corr_heatmap,
                        logger=self.logger)
            
            path_corr_matrix.parent.mkdir(parents=True, exist_ok=True)
            path_corr_metadata.parent.mkdir(parents=True, exist_ok=True)

            corr_matrix.to_parquet(path_corr_matrix)
            corr_metadata.to_parquet(path_corr_metadata)
                
        mlflow.log_artifact(path_corr_matrix, artifact_path=correlation_path_mlflow)
        mlflow.log_artifact(path_corr_metadata, artifact_path=correlation_path_mlflow)

    def _run_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        with mlflow.start_run(run_name=self.config[Stage.FEATURE_ENGINEERING]['run_name'], 
                                  nested=True):
            target_column = self.config[Stage.CLEAN]['target_column']
            train_path_feature_engineered_mlflow = self.config[Stage.FEATURE_ENGINEERING]['train_path_feature_engineered_mlflow']
            test_path_feature_engineered_mlflow = self.config[Stage.FEATURE_ENGINEERING]['test_path_feature_engineered_mlflow']
            train_path_fe = self.project_root / self.config[Stage.FEATURE_ENGINEERING]['train_path_feature_engineered']
            test_path_fe =  self.project_root / self.config[Stage.FEATURE_ENGINEERING]['test_path_feature_engineered']
            remove_correlated_features = bool(self.config[Stage.FEATURE_ENGINEERING]['remove_correlated_features'])
            threshold_correlation = self.config[Stage.FEATURE_ENGINEERING]['threshold_correlation']
            self.logger.info(f'Removing correlated features: {remove_correlated_features}')

                
            self.logger.info(f'Correlation Threshold: {threshold_correlation}')
            mlflow.log_param("correlation_threshold", threshold_correlation)
            mlflow.log_param("remove correlated features", remove_correlated_features)
            features_to_remove = find_correlated_features(X_train, threshold_correlation)
            if features_to_remove:
                features_to_remove_path =  self.project_root / self.config[Stage.FEATURE_ENGINEERING]['correlated_features_path']
                pd.DataFrame(features_to_remove, columns = 'feature').to_parquet(features_to_remove_path)
                mlflow.log_artifact(features_to_remove_path, 
                    artifact_path=self.config['feature_engineering']['correlated_features_path_mlflow'])
                
            train_df_fe , test_df_fe = apply_feature_engineering(X_train, 
                    X_test, 
                    target_column, 
                    remove_correlated_features, 
                    threshold_correlation,
                    features_to_remove, 
                    logger=self.logger)
            
            train_path_fe.parent.mkdir(parents=True, exist_ok=True)
            test_path_fe.parent.mkdir(parents=True, exist_ok=True)

            train_df_fe.to_parquet(train_path_fe)
            test_df_fe.to_parquet(test_path_fe)
        mlflow.log_artifact(train_path_fe, artifact_path=train_path_feature_engineered_mlflow)
        mlflow.log_artifact(test_path_fe, artifact_path=test_path_feature_engineered_mlflow)

    def _run_training(self, X_train: pd.DataFrame,
                            X_test: pd.DataFrame):

        with mlflow.start_run(run_name=self.config[Stage.MODEL_TRAINING]['run_name'], 
                                  nested=True):
            target_column = self.config[Stage.CLEAN]['target_column'] 
            model_name = self.config[Stage.MODEL_TRAINING]['model_name']

            hyperparameters = {}
            hyperparameters['random_state'] = self.config[Stage.MODEL_TRAINING]['random_state']
            hyperparameters['l2_leaf_reg'] = self.config[Stage.MODEL_TRAINING]['l2_leaf_reg']
            hyperparameters['depth'] = self.config[Stage.MODEL_TRAINING]['depth']
            hyperparameters['iterations'] = self.config[Stage.MODEL_TRAINING]['iterations']
            hyperparameters['learning_rate'] = self.config[Stage.MODEL_TRAINING]['learning_rate']

            classification_threshold = float(self.config[Stage.MODEL_TRAINING]['threshold'])
            artifact_path_model_training = self.project_root / self.config[Stage.MODEL_TRAINING]['artifacts_path_model_training']
            registered_model_name = self.config[Stage.MODEL_TRAINING]['registered_model_name']

            artifact_path_model_training.parent.mkdir(parents=True, exist_ok=True)

            results = train_model(
                train_df = X_train,
                test_df = X_test,
                target_column = target_column,
                hyperparameters = hyperparameters,
                threshold = classification_threshold,
                registered_model_name = registered_model_name,
                logger = self.logger)

            mlflow.log_param('classification_threshold', results['Threshold'])
            mlflow.log_param('model_name', model_name)

            mlflow.log_metric('AUC', results['AUC'])
            mlflow.log_metric('Recall', results['Recall'])
            mlflow.log_metric('Precision', results['Precision'])
                
            mlflow.log_param('classification_threshold', results['Threshold'])
            mlflow.log_param('model_name', model_name)

    def _model_evaluation(self, X_test):
        with mlflow.start_run(run_name=self.config[Stage.EVALUATION]['run_name'], 
                    nested=True):
                
            artifacts_path_mlflow_model = self.config[Stage.MODEL_TRAINING]['artifacts_path_mlflow_model_training']
            registered_model_name = self.config[Stage.MODEL_TRAINING]['registered_model_name']

            model, threshold = load_latest_model_and_threshold(
                registered_model_name = registered_model_name,
                artifact_path_mlflow = artifacts_path_mlflow_model,
                file_name_threshold = self.config[Stage.MODEL_TRAINING]['classification_threshold_file_name']
                )

             
               
            target_column = self.config[Stage.CLEAN]['target_column']
            predictions_path = self.project_root / self.config[Stage.EVALUATION]['artifacts_path_inference']
            artifacts_path_predicitions_mlflow = self.config[Stage.EVALUATION]['artifacts_path_inference_mlflow']
            self.logger.info('Generating Predictions')
            predictions = predict(model = model,
                                    data = X_test,
                                    threshold = threshold,
                                    target_column = target_column)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
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
    
    
if __name__ == '__main__':
    main()
