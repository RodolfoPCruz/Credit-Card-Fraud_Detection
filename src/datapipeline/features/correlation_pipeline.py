import pandas as pd
import numpy as np
from typing import List

def find_correlated_features(
    df: pd.DataFrame,
    threshold: float
    ) -> List[str]:

    """
    Identifies highly correlated features based on an absolute
    Pearson correlation threshold.

    This function is intended for use in production pipelines and
    should be applied ONLY to the training dataset.

    Args:
        df (pd.DataFrame): Training dataset 
        threshold (float): correlation threshold above which one 
                        of the features will be removed
    
    Returns:
        List[str]: List of feature names to be removed
    """

    if df.empty:
        return []

    # Compute absolute correlation matrix
    corr_matrix = df.corr().abs()

    # Upper triangle mask (exclude self-correlation)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    #Identify features to drop
    to_drop = [feature for feature in upper_triangle if
               np.any(upper_triangle[feature]>threshold)]


    return to_drop