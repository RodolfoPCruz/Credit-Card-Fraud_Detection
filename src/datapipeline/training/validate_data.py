import pandas as pd
import logging

def validate_data(
        df: pd.DataFrame,
        target_column: str,
        min_samples: int = 1000,
        num_classes: int = 2,
        logger: logging.Logger | None = None
) -> None:
    
    """
    Validates if the dataset is valid

    Args:
        df (pd.DataFrame): pandas dataframe after preliminary cleaning
        target_column (str): target column
        min_samples (int, optional): required min number of samples. Defaults to 1000.
        num_classes (int, optional): number of classes in the target. Defaults to 2.
        logger (logging.Logger, optional): Logger instance.
  
    Raises:
        ValueError: the number of samples is less than min_samples
        ValueError: the number of classes is not equal to num_classes
        ValueError: one target class has no samples
    """


    if df.shape[0] < min_samples:
        raise ValueError(f"Dataset must have at least {min_samples} samples")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    class_count = df[target_column].value_counts(dropna=True)

    if len(class_count) != num_classes:
        raise ValueError(f"Expected {num_classes} classes, found {len(class_count)}")
    
    if class_count.min() <= 0:
     raise ValueError("One target class has no samples")