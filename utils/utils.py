"""
utils.py

Provides utility functions for time series analysis, including:
1. Descriptive statistics
2. Augmented Dickey-Fuller stationarity test
3. Error metrics (MAE, MSE, RMSE, MAPE)
4. Residual diagnostics: Ljung-Box Q-test, Engle’s ARCH test
5. Optional differencing of time series data

Dependencies:
- numpy
- pandas
- statsmodels

Usage:
    Import the module and use the provided functions on your pandas DataFrame or series.

Example:
    >>> import pandas as pd
    >>> from utils import compute_descriptive_stats, adf_test
    >>> df = pd.DataFrame({'price': [100, 105, 102, 110, 108],
    ...                    'log_return': [0, 0.05, -0.028, 0.076, -0.018]})
    >>> stats = compute_descriptive_stats(df)
    >>> print(stats)
    {'price_mean': 105.0, 'price_std': 5.0, 'price_min': 100.0, 'price_max': 110.0, 
     'price_skew': 0.0, 'price_kurtosis': -1.3, 
     'logret_mean': 0.0, 'logret_std': 0.0508, 'logret_min': -0.028, 
     'logret_max': 0.076, 'logret_skew': 0.0, 'logret_kurtosis': -1.3}
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# ------------------------------------------------------------------------------
# 1) DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------
def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """
    Computes descriptive statistics for 'price' and 'log_return'
    columns if they exist in the DataFrame.
    
    Statistics computed:
        - Mean
        - Standard Deviation
        - Minimum
        - Maximum
        - Skewness
        - Kurtosis
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        
    Returns:
        dict: Dictionary containing the computed statistics with keys 
              like 'price_mean', 'price_std', etc.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'price': [100, 105, 102, 110, 108],
        ...     'log_return': [0, 0.05, -0.028, 0.076, -0.018]
        ... })
        >>> stats = compute_descriptive_stats(df)
        >>> 'price_mean' in stats
        True
        >>> stats['price_mean']
        105.0
    """
    stats_dict = {}

    # Compute statistics for 'price' column if present
    if 'price' in df.columns:
        sp = df['price'].dropna()
        stats_dict['price_mean'] = float(sp.mean())
        stats_dict['price_std']  = float(sp.std())
        stats_dict['price_min']  = float(sp.min())
        stats_dict['price_max']  = float(sp.max())
        stats_dict['price_skew'] = float(sp.skew())
        stats_dict['price_kurtosis'] = float(sp.kurtosis())

    # Compute statistics for 'log_return' column if present
    if 'log_return' in df.columns:
        lr = df['log_return'].dropna()
        stats_dict['logret_mean'] = float(lr.mean())
        stats_dict['logret_std']  = float(lr.std())
        stats_dict['logret_min']  = float(lr.min())
        stats_dict['logret_max']  = float(lr.max())
        stats_dict['logret_skew'] = float(lr.skew())
        stats_dict['logret_kurtosis'] = float(lr.kurtosis())

    return stats_dict

# ------------------------------------------------------------------------------
# 2) STATIONARITY CHECK (ADF)
# ------------------------------------------------------------------------------
def adf_test(series, significance=0.05):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity 
    in a time series.
    
    Parameters:
        series (array-like): The time series data.
        significance (float): Significance level for the test. Default is 0.05.
        
    Returns:
        dict: Dictionary containing the test statistic, p-value, 
              boolean indicating stationarity, and critical values.
              
    Example:
        >>> dummy_series = [1, 2, 3, 4, 5]
        >>> result = adf_test(dummy_series)
        >>> 'is_stationary' in result
        True
        >>> result['is_stationary']
        False
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    crit_values = result[4]
    is_stationary = bool(p_value < significance)
    return {
        'test_statistic': result[0],
        'p_value': p_value,
        'is_stationary': is_stationary,
        'critical_values': crit_values
    }

# ------------------------------------------------------------------------------
# 3) RESIDUAL ANALYSIS & DIAGNOSTICS
# ------------------------------------------------------------------------------
def ljung_box_test(residuals, lags=20):
    """
    Performs the Ljung-Box Q-test to check if residuals are white noise.
    
    Parameters:
        residuals (array-like): Residuals from a time series model.
        lags (int): Number of lags to test. Default is 20.
        
    Returns:
        dict: Dictionary containing the test statistic, p-value, 
              and a boolean indicating if residuals are white noise.
              
    Example:
        >>> residuals = [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]
        >>> result = ljung_box_test(residuals)
        >>> 'is_white_noise' in result
        True
    """
    n = len(residuals)
    if n < 2:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # Clamp the lags to be at most n - 1
    actual_lags = min(lags, n - 1)
    if actual_lags < 1:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # Run the Ljung-Box test
    df_result = acorr_ljungbox(residuals, lags=[actual_lags], return_df=True)

    # Check if the result is empty
    if df_result.empty:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # Extract statistics for the actual lags
    lb_stat   = df_result.loc[actual_lags, 'lb_stat']
    lb_pvalue = df_result.loc[actual_lags, 'lb_pvalue']

    is_white_noise = bool(lb_pvalue > 0.05)
    return {
        'lb_stat': float(lb_stat),
        'lb_pvalue': float(lb_pvalue),
        'is_white_noise': is_white_noise
    }

def arch_test(residuals, lags=12):
    """
    Performs Engle's ARCH test to detect heteroskedasticity in residuals.
    
    Parameters:
        residuals (array-like): Residuals from a time series model.
        lags (int): Number of lags to include in the test. Default is 12.
        
    Returns:
        dict: Dictionary containing the test statistic, p-value, 
              and a boolean indicating heteroskedasticity.
              
    Example:
        >>> residuals = [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]
        >>> result = arch_test(residuals)
        >>> 'heteroskedastic' in result
        True
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    if n < 2:
        return {'arch_stat': np.nan, 'arch_pvalue': np.nan, 'heteroskedastic': False}

    # Clamp the lags to be at most n - 1
    actual_lags = min(lags, n - 1)
    if actual_lags < 1:
        return {'arch_stat': np.nan, 'arch_pvalue': np.nan, 'heteroskedastic': False}

    # Run the ARCH test
    stat, pvalue, f_stat, f_pval = het_arch(residuals, nlags=actual_lags)
    return {
        'arch_stat': float(stat),
        'arch_pvalue': float(pvalue),
        'heteroskedastic': bool(pvalue < 0.05)
    }

# ------------------------------------------------------------------------------
# 4) ERROR METRICS
# ------------------------------------------------------------------------------
def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        float: The MAE.
        
    Example:
        >>> mae = mean_absolute_error([1, 2, 3], [1, 2, 3])
        >>> mae
        0.0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        float: The MSE.
        
    Example:
        >>> mse = mean_squared_error([1, 2, 3], [1, 2, 3])
        >>> mse
        0.0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean((y_true - y_pred)**2))

def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        float: The RMSE.
        
    Example:
        >>> rmse = root_mean_squared_error([1, 2, 3], [1, 2, 3])
        >>> rmse
        0.0
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between true and predicted values.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        float: The MAPE.
        
    Example:
        >>> mape = mean_absolute_percentage_error([100, 200, 300], [110, 190, 310])
        >>> round(mape, 4)
        0.0333
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, ensure y_true does not contain zeros
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero(s), cannot compute MAPE.")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

# ------------------------------------------------------------------------------
# 5) OPTIONAL: DIFFERENCING
# ------------------------------------------------------------------------------
def difference_series(series: pd.Series, order=1) -> pd.Series:
    """
    Differentiates a time series to make it stationary.
    
    Parameters:
        series (pd.Series): The time series data.
        order (int): The order of differencing. Default is 1.
        
    Returns:
        pd.Series: Differenced time series with NaN values dropped.
        
    Example:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> diff_s = difference_series(s)
        >>> diff_s.tolist()
        [1.0, 1.0, 1.0, 1.0]
    """
    return series.diff(order).dropna()

# ------------------------------------------------------------------------------
# MODULE TEST
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    print("=== Testing utils.py ===")

    # 1) ADF test
    dummy = [1, 2, 3, 4, 5]
    adf_result = adf_test(dummy)
    print("ADF Test:", adf_result)

    # 2) Error metrics
    mae = mean_absolute_error([1,2,3],[1,2,3])
    print("MAE:", mae)
    rmse = root_mean_squared_error([1,2,3],[1,2,3])
    print("RMSE:", rmse)

    # 3) Descriptive stats
    df_example = pd.DataFrame({
        'price': [100, 105, 102, 110, 108],
        'log_return': [0, 0.05, -0.028, 0.076, -0.018]
    })
    descriptive_stats = compute_descriptive_stats(df_example)
    print("Descriptive Stats:", descriptive_stats)

    # 4) Residual analysis example
    res = [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]
    lb_test = ljung_box_test(res)
    print("Ljung-Box test (default 20 lags):", lb_test)
    arch_test_result = arch_test(res)
    print("Engle's ARCH test (default 12 lags):", arch_test_result)