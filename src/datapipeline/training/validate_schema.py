import pandas as pd


def validate_schema(df: pd.DataFrame, schema: dict):

    """
    Verifyt if the dataframe matches the schema

    Args:
        df (pd.DataFrame): pandas dataframe containing the raw dataset
        schema (dict): dictionary containing the schema

    Raises:
        ValueError: there is a missing or extra column
        TypeError: the target column is not numeric
        
    """

    target_column = schema['target_column']
    
    #---Expected columns
    expected_features = set()
    for feature in schema['features'].values():
        expected_features.update(feature)
        

    expected_columns = expected_features | {target_column}
    df_columns = set(df.columns)

    missing = expected_columns - df_columns
    extra = df_columns - expected_columns

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if extra and not schema['allow_extra_columns']:
        raise ValueError(f"Unexpected columns: {extra}")


    #---Numeric Features
    for feature in schema['features']['numeric']:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f'The feature {feature} must be numeric')
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(f"Target column '{target_column}' must be numeric")


    #---missing values in the target column
    if df[target_column].isna().any():
        raise ValueError(
        f"Target column '{target_column}' contains missing values"
         )

    #---Unique values in the target column
    expected_nunique = schema['nunique_target_column']
    nunique = df[target_column].nunique()
    if nunique != expected_nunique:
        raise ValueError(
            f"Target must have {expected_nunique} unique values, not {nunique}"
        )
