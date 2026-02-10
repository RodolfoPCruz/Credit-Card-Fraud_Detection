import pandas as pd
import numpy as np


def predict(
        model,
        data: pd.DataFrame,
        threshold: float,
        target_column: str = None
) -> pd.DataFrame:
    """
    Predicts the class labels for a given dataset using a trained model and a
    classification threshold.

    Args:   
        model (catboost.CatBoostClassifier): Trained CatBoost model
        data (pd.DataFrame): Input dataset
        threshold (float): Classification threshold
        target_column (str): Name of the target column. None if the target
                             column is not present

    Returns:
        pd.DataFrame: DataFrame containing the predicted class labels
                      and predicted probabilities   
    """

    if target_column is not None:
        data = data.drop(target_column, axis=1)
    
    probs = model.predict_proba(data)[:,1]
    predictions = (probs >= threshold).astype(int)

    return pd.DataFrame({
        "prediction": predictions,
        "probability": probs
    })
           