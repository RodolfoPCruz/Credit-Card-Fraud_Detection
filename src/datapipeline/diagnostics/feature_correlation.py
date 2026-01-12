import pandas as pd
import scipy as sp
import numpy as np
from scipy.stats import (skew, 
                        kurtosis, 
                        shapiro, 
                        pearsonr, 
                        normaltest, 
                        norm, 
                        spearmanr,
                        pointbiserialr,
                        kendalltau)
#import matplotlib.pyplot as plt
from scipy.stats.contingency import association


from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_integer_dtype
    
)


class FeatureCorrelation:
    """
    Performs an exploratory correlation analysis between features of a pandas 
    DataFrame.

    This class automatically:
    - infers feature types (continuous, binary, ordinal, categorical, etc.)
    - selects an appropriate correlation method based on feature types
      and distributional assumptions
    - computes correlation coefficients and optional statistical metadata

    The class is intended for exploratory data analysis (EDA) and
    diagnostics, not for production feature selection pipelines.

    Parameters:
        df (pd.DataFrame): Input dataframe containing features only.
        alpha (float): Significance level used for normality tests.
        max_discrete_unique (int): Maximum number of unique values for a
            numeric feature to be considered discrete.
        ordinal_map (dict, optional): Mapping for ordinal categorical 
        variables.
    """

    def __init__(self,
                  df: pd.DataFrame, 
                  alpha: float = 0.05,
                  max_discrete_unique: int = 10, 
                  ordinal_map: dict | None = None):
        """
        Initializes the FeatureCorrelation analyzer and caches feature types.
        """
        self.df = df
        self.alpha = alpha
        self.max_discrete_unique = max_discrete_unique
        self.ordinal_map = ordinal_map or {}
         
        # Cache detected feature types
        self.feature_types = {
            col: self.detect_type(self.df[col]) for col in self.df.columns
         }

    # --------------------
    # Feature type detection
    # --------------------
    def detect_type(self, series: pd.Series) -> str:
       """
       Infers the semantic/statistical type of a pandas Series based on
       dtype, cardinality, and optional ordinal mapping.

       """
       series = series.dropna()

       if series.empty:
           return 'unknown'
       
       #boolean
       if is_bool_dtype(series):
           return 'binary'
       
       #datetime
       if is_datetime64_any_dtype(series):
           return 'datetime'
       
       if is_numeric_dtype(series):
           unique_vals = series.unique()

           #binary
           if set(unique_vals) == {0,1}:
               return 'binary'

           #discrete numeric
           if (
               is_integer_dtype(series) 
               and len(unique_vals) <= self.max_discrete_unique
           ):
               return 'discrete_numeric'

           return 'continuous'
       
       if is_categorical_dtype(series) or is_object_dtype(series):
           
           if self.ordinal_map is not None:
               if set(series.unique()).issubset(set(self.ordinal_map.keys())):
                   return 'ordinal'
           
           return 'categorical'
       
       return 'unknown'
    
    # --------------------
    # Normality check
    # --------------------
    def check_normality(self, series: pd.Series) -> bool:
        """
        Evaluates whether a numeric series can be considered approximately 
        normal.

        The decision is based on:
            - sample size (Shapiro-Wilk for n < 5000, 
                D'Agostino-Pearson otherwise)
            - skewness and kurtosis thresholds
            - significance level defined by alpha


        Returns:
            bool: True if the distribution is considered approximately normal.
        """

        series = series.dropna()

        if len(series) < 8:
            return False

        skewness = skew(series)
        kurt = kurtosis(series, fisher=True)

        n_samples = len(series)

        if n_samples<5000:
            _, p_value = shapiro(series)
        else:
            _, p_value = normaltest(series)

        if (p_value < self.alpha or 
            abs(skewness) > 1 or 
            abs(kurt) > 1):
            return False
       
        return True
    
    # --------------------
    # Correlation method selection
    # --------------------
    def choose_correlation(self, 
                           col_x: pd.Series, 
                           col_y: pd.Series) -> str:
        """
        Selects an appropriate correlation method based on the inferred
        types and distributional properties of two features.

        Possible methods include Pearson, Spearman, Kendall's Tau,
        and Cramér's V.

        Returns:
            str: Name of the selected correlation method, or 'not_applicable'
            if no suitable method is available.
        """
        tx = self.detect_type(col_x)
        ty = self.detect_type(col_y)    

        if tx == ty == 'continuous':
            if (self.check_normality(col_x) 
                and self.check_normality(col_y)):
                return 'pearson'
            return 'spearman'
            
        if tx == ty == 'binary':
            return 'pearson'
        
        if {"binary", "continuous"} == {tx, ty}:
            return 'pearson' 
            
        if tx == ty == 'discrete_numeric':
            return 'spearman' 

        if tx == ty == 'categorical':
            return 'cramers_v'

        if tx == ty == 'ordinal':
            return 'kendall_tau_b'
        
        if {"ordinal", "continuous"} == {tx, ty}:
            return "spearman"

        return 'not_applicable'
    
    # ------------------------------------------------------------------
    # Correlation computation (numeric only)
    # ------------------------------------------------------------------
    def calculate_corelation(self,
                              col_x: pd.Series, 
                              col_y: pd.Series, 
                              ) -> float:
        
        """
        Computes the correlation coefficient and p-value between two features
        using an automatically selected correlation method.

        Returns:
            Tuple[float, float]: Correlation coefficient and p-value.
            For methods where a p-value is not defined, NaN is returned.
        """
       
        correlation_method = self.choose_correlation(col_x, col_y)

        if correlation_method == 'pearson':
            res = pearsonr(col_x, col_y)
            return res.statistic, res.pvalue
        if correlation_method == 'spearman':
            res = spearmanr(col_x, col_y)
            return res.statistic, res.pvalue
        if correlation_method == 'kendall_tau_b':
            res = kendalltau(col_x, col_y)
            return res.statistic, res.pvalue
        if correlation_method == 'cramers_v':
            contigency_table = pd.crosstab(col_x, col_y)
            return association(contigency_table), np.nan
        return np.nan, np. nan
    
    # ------------------------------------------------------------------
    # Correlation matrix (numeric only)
    # ------------------------------------------------------------------
    
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Computes a symmetric correlation matrix for all features
        in the dataframe.
        """
        
        features = self.df.columns
        corr_df = pd.DataFrame(np.nan,
                                index=features,
                                  columns=features)
        
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if j < i:
                    corr_df.loc[f1, f2] = corr_df.loc[f2, f1]
                elif i == j:
                    corr_df.loc[f1, f2] = 1.0
                else:
                    corr, _ = self.calculate_corelation(
                        self.df[f1], self.df[f2]
                    )
                    corr_df.loc[f1, f2] = corr
                    corr_df.loc[f2, f1] = corr
        return corr_df
    
    # ------------------------------------------------------------------
    # Correlation + metadata
    # ------------------------------------------------------------------

    def calculate_correlation_with_metadata(
        self, 
        col_x: pd.Series,
        col_y: pd.Series,
    ) -> dict:

        """
        Computes the correlation between two features along with
        diagnostic metadata such as feature types, normality flags,
        selected method, and sample size.

        Returns:
            dict: Dictionary containing correlation statistics and metadata.
        """

        valid_data = pd.concat([col_x, col_y], axis = 1).dropna()
        x = valid_data.iloc[:, 0]
        y = valid_data.iloc[:, 1]

        tx = self.detect_type(x)
        ty = self.detect_type(y)

        normal_x = self.check_normality(x) if tx == 'continuous' else None
        normal_y = self.check_normality(y) if ty == 'continuous' else None

        correlation_method = self.choose_correlation(x, y)
        
        if correlation_method == 'pearson':
            corr, p_value = pearsonr(x, y)
        elif correlation_method == 'spearman':
            corr, p_value = spearmanr(x, y)
        elif correlation_method == 'kendall_tau_b':
            corr, p_value = kendalltau(x, y)
        elif correlation_method == 'cramers_v':
            contigency_table = pd.crosstab(x, y)
            corr = association(contigency_table)
            p_value = np.nan    
        else:
            corr = np.nan
            p_value = np.nan

        return {
            'method': correlation_method,
            'correlation': corr,
            'p_value': p_value,
            'type_x': tx,
            'type_y': ty,
            'normal_x': normal_x,
            'normal_y': normal_y,
            'n_samples': len(valid_data)
        }
        
    
    # ------------------------------------------------------------------
    # Correlation + metadata
    # ------------------------------------------------------------------
      
    def correlation_metadata_table(self) -> pd.DataFrame:
        """
        Generates a table containing correlation statistics and metadata
        for all unique feature pairs.

        Returns:
            pd.DataFrame: Long-format table with one row per feature pair.
        """
        records = []
        features = self.df.columns

        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if j <= i:
                    continue

                result = self.calculate_correlation_with_metadata(
                    self.df[f1], self.df[f2]
                )

                records.append({
                    "feature_x": f1,
                    "feature_y": f2,
                    **result
                })

        return pd.DataFrame(records)
    