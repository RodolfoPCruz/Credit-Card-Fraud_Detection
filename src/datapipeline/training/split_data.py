import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

def split_data(df: pd.DataFrame,
               target_column: str,
               test_size: float,
               random_state: int,
               logger: logging.Logger | None = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Split a DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        logger (logging.Logger | None, optional): The logger to use. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if logger:
        logger.info('Spliting data...')

    if y.nunique() < 2:
        raise ValueError("Target column must have at least two classes for stratified split")

        
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        stratify=y)
    if logger:
        logger.info(f'{df.shape[0]} samples split into {X_train.shape[0]} ' 
                    f'train samples and {X_test.shape[0]} test samples')
        logger.info(f'{X_train.shape[0]/df.shape[0]*100:.2f}% of the '
                    f'dataset is used for training and {X_test.shape[0]/df.shape[0]*100:.2f}% for testing' )

    df_train = X_train
    df_test  = X_test
    df_train[target_column] = y_train
    df_test[target_column] = y_test

    return df_train, df_test