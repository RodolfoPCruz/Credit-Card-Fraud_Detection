from datapipeline.diagnostics.feature_correlation import FeatureCorrelation
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_correlation_diagnostics(df: pd.DataFrame,
               logger: logging.Logger | None = None
               ):
    """
    Calculates correlation between each pair of geatures in a dataframe. It 
    also genetes a table table containing correlation statistics and metadata
    for all unique feature pairs

    Args:
        df (pd.DataFrame): pandas dataframe containing the features to be 
            analysed.
        path_savefig (str): path to save plot of correlation matrix. 
        logger (logging.Logger | None, optional): The logger to use. 
            Defaults to None.

    Returns:
        correlation_matrix (pd.DataFrame): a dataframe containing the 
                calculated correlation between each pair of features 
        correlation_metadata (pd.DataFrame):  dataframe containing correlation 
                statistics and metadata for all unique feature pairs
    """
    corr = FeatureCorrelation(df)

    if logger:
        logger.info('Performing correlation analysis')

    correlation_matrix = corr.correlation_matrix()
    correlation_metadata = corr.correlation_metadata_table()
	
	
    return (correlation_matrix, 
            correlation_metadata)
