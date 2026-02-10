import pandas as pd
import logging

def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    tolerance: float = 0.01,
    logger: logging.Logger | None = None
)-> None:
    
    """
    Validates if the split is valid.

    Args:
        train_df (pd.DataFrame): training data  
        test_df (pd.DataFrame): testing dataset
        target_column (str): target column
        tolerance (float, optional): Class distribution shift tolerance. The difference between 
                                    train and test class distribution must be less than this . Defaults to 0.01.
        logger (logging.Logger, optional): Logger instance.

    Raises:
        ValueError: _description_
    """
    

    train_ratio = train_df[target_column].mean()
    test_ratio = test_df[target_column].mean()

    diff = abs(train_ratio - test_ratio)

    if diff > tolerance:
        raise ValueError(
            f"Class distribution shift too large: {diff:.4f}"
        )

    if logger:
        logger.info(
            f"Split validation passed "
            f"(train={train_ratio:.4f}, test={test_ratio:.4f})"
        )
