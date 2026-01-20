from datapipeline.features.correlation_pipeline import find_correlated_features
import pandas as pd 
import logging

def apply_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    remove_correlated: bool,
    correlation_threshold: float,
    correlated_features: pd.DataFrame = None,
    logger: logging.Logger | None = None
    ):
    """
    Removes correlated features based on an absolute Pearson 
        correlation threshold. For a pair of features, if they 
        are highly correlated, the first that appears in the 
        dataframe will be removed

    Args:
        train_df (pd.DataFrame): training dataset
        test_df (pd.DataFrame): testing dataset
        target_column (str): target column of the dataset
        remove_correlated (bool): whther to remove correlated features
        correlation_threshold (float): threshold above which one of 
            the features will be removed
        correlated_features (list): list of correlated features
        logger (logging.Logger | None, optional): logger object

    Returns:
        train_df_fe (pd.DataFrame): feature engineered training dataset
        test_df_fe (pd.DataFrame): feature engineered testing dataset
    """

    X_train = train_df.drop(columns = target_column)
    X_test = test_df.drop(columns = target_column)

    if remove_correlated:
        if correlated_features is None:
            correlated_features = find_correlated_features(X_train, correlation_threshold)
        X_train = X_train.drop(columns = correlated_features)
        X_test = X_test.drop(columns = correlated_features)

        if logger:
            logger.info(f'Correlated features removed: {correlated_features}')

    train_df_fe = pd.concat([X_train, train_df[target_column]], axis=1)
    test_df_fe = pd.concat([X_test, test_df[target_column]], axis=1)
    
    return train_df_fe, test_df_fe
