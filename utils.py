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
        The time-series data (e.g., log returns).
    significance : float
        Significance level for determining stationarity.

    Returns
    -------
    dict
        A dictionary containing:
        - test_statistic
        - p_value
        - is_stationary (bool)

    Examples
    --------
    >>> result = adf_test([1,2,3,4,5])
    >>> 'p_value' in result
    True
    """
    # Run the ADF test using statsmodels
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

    MAE = average of |y_true - y_pred|

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        MAE value.

    Examples
    --------
    >>> mean_absolute_error([1.0, 2.0, 3.0], [1.1, 1.9, 3.0])
    0.06666666666666671
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE).

    MSE = average of (y_true - y_pred)^2

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        MSE value.

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

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        RMSE value.

    Examples
    --------
    >>> root_mean_squared_error([1,2,3],[1,2,3])
    0.0
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    MAPE = average of ( |y_true - y_pred| / y_true )

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        MAPE value, e.g. 0.10 means 10% error.

    Examples
    --------
    >>> mean_absolute_percentage_error([100,200,300],[90,210,310])
    0.03333333333333333
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
