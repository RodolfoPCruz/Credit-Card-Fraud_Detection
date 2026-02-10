from catboost import CatBoostClassifier
import logging
import pandas as pd
from sklearn.metrics import (precision_recall_curve, 
                            auc,
                            precision_score,
                            recall_score)
import mlflow
from mlflow.models import infer_signature
import json
from pathlib import Path

def train_model(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                target_column: str,
                hyperparameters: dict,
                threshold: float,
                registered_model_name: str,
                artifacts_dir: str,
                artifacts_path_mlflow: str,
                threshold_file_name: str,
                logger: logging.Logger | None = None
                ) -> dict:
    """
    Trains a model using CatBoostClassifier and logs the model to mlflow

    Args:
        train_df (pd.DataFrame): training dataset
        test_df (pd.DataFrame): testing dataset
        target_column (str): target column of the dataset
        hyperparameters (dict): dictionary of hyperparameters 
        threshold (float): classification threshold
        logger (logging.Logger | None, optional): The logger to use. Defaults to None.
    Returns
        results (dict): dictionary of model metrics
    """

    mlflow.set_tag("model", "catboost")
    mlflow.set_tag("task", "fraud_detection")


    mlflow.log_params(hyperparameters)

    if logger:
        logger.info('Training model...')

    catboost = CatBoostClassifier(**hyperparameters)

    y_train = train_df[target_column]
    x_train = train_df.drop(columns=[target_column])

    y_test = test_df[target_column]
    x_test = test_df.drop(columns=[target_column])

    catboost.fit(x_train, y_train, verbose = False)

    if logger:
        logger.info('Model trained')

    threshold_value = threshold
    threshold_dict = {"threshold": threshold_value}
    y_pred_proba = catboost.predict_proba(x_test)[:,1]
    y_pred = y_pred_proba > threshold_value

    signature = infer_signature(x_train, y_pred_proba)

    mlflow.set_tag("classification_threshold", threshold_value)
    mlflow.catboost.log_model(catboost, 
                              "model",
                              signature=signature,
                              registered_model_name=registered_model_name,
                              input_example=x_train.iloc[:5])

    file_name = threshold_file_name
    artifacts_dir = Path(artifacts_dir)
    with open(artifacts_dir / file_name, 'w') as f:
        json.dump(threshold_dict, f)
    mlflow.log_artifact(artifacts_dir / file_name, 
                        artifact_path=artifacts_path_mlflow)


    precision, recall, thresholds = precision_recall_curve(
    y_test, y_pred_proba)
   
    auc_score = auc(recall, precision)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    if logger:
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"AUC: {auc_score}")

    results = {
        'Precision': precision,
        'Recall': recall,
        'AUC': auc_score,
        'Threshold': threshold
    }

    return results
    