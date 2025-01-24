"""
utils.py

Provides:
1. Descriptive statistics function (compute_descriptive_stats)
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
    columns if they exist in the DataFrame. Returns a dictionary
    of stats like { 'price_mean', 'price_std', 'price_min', etc. }.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns might include ['price', 'log_return'].

    Returns
    -------
    dict
        e.g. {
          'price_mean': float,
          'price_std': float,
          'price_min': float,
          'price_max': float,
          'price_skew': float,
          'price_kurtosis': float,
          'logret_mean': float,
          'logret_std': float,
          ... etc ...
        }
    """
    stats_dict = {}

    if 'price' in df.columns:
        ser_price = df['price'].dropna()
        stats_dict['price_mean'] = float(ser_price.mean())
        stats_dict['price_std'] = float(ser_price.std())
        stats_dict['price_min'] = float(ser_price.min())
        stats_dict['price_max'] = float(ser_price.max())
        stats_dict['price_skew'] = float(ser_price.skew())
        stats_dict['price_kurtosis'] = float(ser_price.kurtosis())

    if 'log_return' in df.columns:
        ser_lr = df['log_return'].dropna()
        stats_dict['logret_mean'] = float(ser_lr.mean())
        stats_dict['logret_std'] = float(ser_lr.std())
        stats_dict['logret_min'] = float(ser_lr.min())
        stats_dict['logret_max'] = float(ser_lr.max())
        stats_dict['logret_skew'] = float(ser_lr.skew())
        stats_dict['logret_kurtosis'] = float(ser_lr.kurtosis())

    return stats_dict


# ------------------------------------------------------------------------------
# 2) STATIONARITY CHECK: ADF TEST
# ------------------------------------------------------------------------------
def adf_test(series, significance=0.05):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters
    ----------
    series : array-like
        Time-series data (e.g., log returns or price).
    significance : float
        Significance level for concluding stationarity. Default=0.05

    Returns
    -------
    dict
        {
            'test_statistic': float,
            'p_value': float,
            'is_stationary': bool,
            'critical_values': dict
        }
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    crit_values = result[4]  # dictionary of critical values
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
    Ljung-Box Q-test to check if residuals are white noise.
    
    Parameters
    ----------
    residuals : array-like
    lags : int, optional
        Number of lags to test. Default=20.

    Returns
    -------
    dict
        {
          'lb_stat': float (test statistic),
          'lb_pvalue': float,
          'is_white_noise': bool (True if p>0.05 => fails to reject random noise)
        }
    """
    # acorr_ljungbox returns arrays; we take the last values for the highest lag
    lb_results = acorr_ljungbox(residuals, lags=[lags], return_df=False)
    lb_stat = lb_results[0][-1]   # test statistic at lag=20
    lb_pvalue = lb_results[1][-1] # p-value
    is_white_noise = (lb_pvalue > 0.05)
    return {
        'lb_stat': lb_stat,
        'lb_pvalue': lb_pvalue,
        'is_white_noise': is_white_noise
    }


def arch_test(residuals, lags=1):
    """
    Engle's ARCH test for heteroskedasticity in residuals.

    Parameters
    ----------
    residuals : array-like
    lags : int, optional
        Number of lags to test. Default=1.

    Returns
    -------
    dict
        {
          'arch_stat': float (test statistic),
          'arch_pvalue': float,
          'heteroskedastic': bool (True if p<0.05 => there's heteroskedasticity)
        }
    """
    # het_arch returns (LM-test statistic, p-value, f-statistic, f p-value)
    stat, pvalue, _, _ = het_arch(residuals, lags=lags)
    return {
        'arch_stat': stat,
        'arch_pvalue': pvalue,
        'heteroskedastic': bool(pvalue < 0.05)
    }


# ------------------------------------------------------------------------------
# 4) ERROR METRICS
# ------------------------------------------------------------------------------
def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# ------------------------------------------------------------------------------
# 5) OPTIONAL: DIFFERENCING IF NON-STATIONARY
# ------------------------------------------------------------------------------
def difference_series(series: pd.Series, order=1) -> pd.Series:
    """
    Applies simple differencing to help achieve stationarity if needed.

    Parameters
    ----------
    series : pd.Series
        Original time-series data.
    order : int
        Order of differencing. Default=1 => (y[t] - y[t-1]).

    Returns
    -------
    pd.Series
        Differenced series with one less data point for each differencing order.
    """
    return series.diff(order).dropna()


# ------------------------------------------------------------------------------
# MODULE TEST
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demonstration

    # 1) ADF test
    dummy = [1, 2, 3, 4, 5]
    print("ADF Test:", adf_test(dummy))

    # 2) Error metrics
    print("MAE:", mean_absolute_error([1,2,3],[1,2,3]))
    print("RMSE:", root_mean_squared_error([1,2,3],[1,2,3]))

    # 3) Example of descriptive stats
    df_example = pd.DataFrame({
        'price': [100, 105, 102, 110, 108],
        'log_return': [0, 0.05, -0.028, 0.076, -0.018]
    })
    print("Descriptive Stats:", compute_descriptive_stats(df_example))

    # 4) Residual analysis example
    res = [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]
    print("Ljung-Box test:", ljung_box_test(res))
    print("Engle's ARCH test:", arch_test(res))
