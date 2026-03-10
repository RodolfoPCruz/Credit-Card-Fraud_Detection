from catboost import CatBoostClassifier
import logging
import pandas as pd
from sklearn.metrics import (precision_recall_curve, 
                            auc,
                            precision_score,
                            recall_score)
import mlflow
from mlflow.models import infer_signature
from pathlib import Path
import mlflow.pyfunc
import os
import tempfile
import joblib


class ModelWithThreshold(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])
        self.threshold = joblib.load(context.artifacts["threshold_path"])

    def predict(self, context, model_input):
        probs = self.model.predict_proba(model_input)[:, 1]
        return (probs >= float(self.threshold)).astype(int)
        
def train_model(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                target_column: str,
                hyperparameters: dict,
                threshold: float,
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
    mlflow.set_tag("classification_threshold", threshold)


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

    
    y_pred_proba = catboost.predict_proba(x_test)[:,1]
    y_pred = y_pred_proba > threshold

    signature = infer_signature(x_train, y_pred_proba)

    mlflow.set_tag("classification_threshold", threshold)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "catboost.pkl")
        threshold_path = os.path.join(tmp_dir, "threshold.pkl")

        joblib.dump(catboost, model_path)
        joblib.dump(threshold, threshold_path)

        mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ModelWithThreshold(),
        artifacts={
            "model_path": model_path,
            "threshold": threshold_path
        },
        signature=signature,
        input_example=x_train.iloc[:5]
    )    


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
    