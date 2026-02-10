import pandas as pd
import logging
import argparse
import yaml   
from typing import Tuple

def clean_data(df: pd.DataFrame, 
               target_column: str, 
               logger: str | None = None
) -> Tuple[pd.DataFrame, int]:

    """
    Perform preliminary data cleaning.

    Steps:
    - Remove duplicate rows
    - Remove rows without target
    - Sanity check on target values

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of target column.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        Tuple[pd.DataFrame, int]: Cleaned dataframe and number of rows removed.
    """

    initial_rows = df.shape[0]

    # Remove duplicates
    df = df.drop_duplicates()
    duplicated_rows_removed = initial_rows - df.shape[0]
    
    if logger:
         logger.info(f'Removed {duplicated_rows_removed} duplicated rows')


    # Remove rows without target
    before = df.shape[0]
    df = df.dropna(subset=[target_column])
    removed_missing_target = before - df.shape[0]
    
    if logger:
        logger.info(f'Removed {removed_missing_target} rows without target')

    # Sanity checks
    if (df[target_column] < 0).any():
        raise ValueError("Invalid target values detected")

    total_removed = duplicated_rows_removed + removed_missing_target
    if logger:
        logger.info(f'Total rows removed: {total_removed}')

    return df, total_removed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_file_path, "r") as f:
            config = yaml.safe_load(f)

    
    