"""
utils.py

Provides utility functions for:
1. Augmented Dickey-Fuller stationarity test
2. Error metrics (MAE, MSE, RMSE, MAPE)

Author: Your Name
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(series, significance=0.05):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters
    ----------
    series : array-like
        Time-series data (e.g., log returns).
    significance : float
        Significance level for concluding stationarity.

    Returns
    -------
    dict
        {
            'test_statistic': float,
            'p_value': float,
            'is_stationary': bool
        }

    Examples
    --------
    >>> result = adf_test([1,2,3,4,5])
    >>> 'p_value' in result
    True
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = (p_value < significance)
    return {
        'test_statistic': result[0],
        'p_value': p_value,
        'is_stationary': is_stationary
    }


def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).

    MAE = mean( abs(y_true - y_pred) )

    Examples
    --------
    >>> mean_absolute_error([1,2,3],[1,2,3])
    0.0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).

    MSE = mean( (y_true - y_pred)^2 )

    Examples
    --------
    >>> mean_squared_error([1,2,3],[1,2,3])
    0.0
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).

    RMSE = sqrt( MSE )

    Examples
    --------
    >>> root_mean_squared_error([1,2,3],[1,2,3])
    0.0
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    MAPE = mean( abs( (y_true - y_pred) / y_true ) )

    Examples
    --------
    >>> mean_absolute_percentage_error([100,200,300],[90,210,310])
    0.03333333333333333
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


if __name__ == "__main__":
    # Simple tests
    print("ADF:", adf_test([1,2,3,4,5]))
    print("MAE:", mean_absolute_error([1,2,3],[1,2,3]))
    print("RMSE:", root_mean_squared_error([1,2,3],[1,2,3]))