"""
utils.py

Provides:
1. Descriptive statistics
2. Augmented Dickey-Fuller stationarity test
3. Error metrics (MAE, MSE, RMSE, MAPE)
4. Residual diagnostics: Ljung-Box Q-test, Engle’s ARCH test
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
    Returns a dictionary with keys like 'price_mean','price_std', etc.
    """
    stats_dict = {}

    if 'price' in df.columns:
        sp = df['price'].dropna()
        stats_dict['price_mean'] = float(sp.mean())
        stats_dict['price_std']  = float(sp.std())
        stats_dict['price_min']  = float(sp.min())
        stats_dict['price_max']  = float(sp.max())
        stats_dict['price_skew'] = float(sp.skew())
        stats_dict['price_kurtosis'] = float(sp.kurtosis())

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
    Augmented Dickey-Fuller test for stationarity.
    Returns dict with test_statistic, p_value, is_stationary, critical_values
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
    Ljung-Box Q-test => check if residuals are white noise.
    
    We clamp lags = min(lags, len(residuals)-1).
    Use return_df=True so statsmodels returns a DataFrame with columns
    ['lb_stat','lb_pvalue','bp_stat','bp_pvalue'] for each tested lag.
    """
    n = len(residuals)
    if n < 2:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # clamp the lags
    actual_lags = min(lags, n - 1)
    if actual_lags < 1:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # run the test
    # This returns a DataFrame with index=the lag used, columns=...
    df_result = acorr_ljungbox(residuals, lags=[actual_lags], return_df=True)

    # if empty, no result
    if df_result.empty:
        return {'lb_stat': np.nan, 'lb_pvalue': np.nan, 'is_white_noise': False}

    # The row is at index 'actual_lags'
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
    Engle's ARCH test for heteroskedasticity in residuals.

    We clamp lags = min(lags, len(residuals)-1).
    If that becomes < 1, we return NaN results. Otherwise, we do het_arch(..., nlags=...).
    """
    # Convert to array if it's a list
    residuals = np.asarray(residuals)
    n = len(residuals)
    if n < 2:
        return {'arch_stat': np.nan, 'arch_pvalue': np.nan, 'heteroskedastic': False}

    # clamp the lags
    actual_lags = min(lags, n - 1)
    if actual_lags < 1:
        return {'arch_stat': np.nan, 'arch_pvalue': np.nan, 'heteroskedastic': False}

    # run the ARCH test
    # For older statsmodels, we must pass 'nlags=actual_lags'
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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean((y_true - y_pred)**2))

def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

# ------------------------------------------------------------------------------
# 5) OPTIONAL: DIFFERENCING
# ------------------------------------------------------------------------------
def difference_series(series: pd.Series, order=1) -> pd.Series:
    return series.diff(order).dropna()

# ------------------------------------------------------------------------------
# MODULE TEST
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing utils.py ===")

    # 1) ADF test
    dummy = [1, 2, 3, 4, 5]
    print("ADF Test:", adf_test(dummy))

    # 2) Error metrics
    print("MAE:", mean_absolute_error([1,2,3],[1,2,3]))
    print("RMSE:", root_mean_squared_error([1,2,3],[1,2,3]))

    # 3) Descriptive stats
    df_example = pd.DataFrame({
        'price': [100, 105, 102, 110, 108],
        'log_return': [0, 0.05, -0.028, 0.076, -0.018]
    })
    print("Descriptive Stats:", compute_descriptive_stats(df_example))

    # 4) Residual analysis example
    res = [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]
    print("Ljung-Box test (default 20 lags):", ljung_box_test(res))       # clamps to 6
    print("Engle's ARCH test (default 12 lags):", arch_test(res))         # clamps to 6
