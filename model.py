"""
model.py

Implements:
1. ARIMA model fitting
2. GARCH model fitting on ARIMA residuals
3. Forecasting function that combines ARIMA mean forecast with GARCH volatility

Author: Your Name
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from typing import Tuple

def fit_arima_garch(
    train_returns: pd.Series, 
    arima_order: Tuple[int,int,int] = None, 
    garch_order: Tuple[int,int] = (1,1),
    dist: str = 'normal'
):
    """
    Fits an ARIMA model to the returns, then fits a GARCH model to the ARIMA residuals.

    Parameters
    ----------
    train_returns : pd.Series
        The log returns series used for training.
    arima_order : Tuple[int,int,int]
        ARIMA hyperparameters (p, d, q). If None, a default (1, 0, 1) is used.
        Ideally, you'd use an auto-ARIMA approach or manual AIC/BIC tuning.
    garch_order : Tuple[int,int]
        GARCH hyperparameters (p, q). By default, GARCH(1,1).
    dist : str
        Distribution assumption for the GARCH residuals. Options: 'normal', 't', 'skewt'.

    Returns
    -------
    (ARIMAResults, ARCHModelResult)
        - Fitted ARIMA model object
        - Fitted GARCH model results

    Examples
    --------
    You can test a short time series manually (though recommended is a real log-return series).
    >>> import pandas as pd
    >>> data = pd.Series([0.01, 0.02, -0.01, 0.015])
    >>> arima_model, garch_res = fit_arima_garch(data, (1,0,1), (1,1), dist='normal')
    """
    # If no ARIMA order is provided, we use a simple default
    if arima_order is None:
        arima_order = (1, 0, 1)

    # 1) Fit ARIMA for the mean of log returns
    arima_model = ARIMA(train_returns, order=arima_order).fit()
    # arima_model.resid provides the residuals (errors) from the ARIMA model

    # 2) Fit GARCH model to the ARIMA residuals
    # The arch_model requires a series; p & q are the GARCH orders
    # dist can be 'normal', 't' (Student's t), or 'skewt'.
    garch = arch_model(arima_model.resid, p=garch_order[0], q=garch_order[1], dist=dist)
    garch_res = garch.fit(disp="off")  # disp="off" to suppress console output

    return arima_model, garch_res


def forecast_arima_garch(
    arima_model, 
    garch_model, 
    steps: int = 30
) -> pd.DataFrame:
    """
    Forecasts future log returns using ARIMA for the mean 
    and GARCH for conditional volatility.

    Parameters
    ----------
    arima_model : ARIMAResults
        Fitted ARIMA model (statsmodels).
    garch_model : ARCHModelResult
        Fitted GARCH model (arch library).
    steps : int
        Number of forecast steps (e.g., 30 days).

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - 'mean_return': forecasted mean of log returns
        - 'volatility': forecasted volatility (std dev) from the GARCH
        * Additional metadata could be added.

    Notes
    -----
    - This function returns the forecasted returns and volatility. You will need
      to convert returns to prices outside this function (by exponentiating 
      and multiplying by the last known price, for instance).

    Examples
    --------
    (see fit_arima_garch's docstring for usage)
    """
    # Forecast the mean of the residuals (ARIMA part)
    arima_forecast = arima_model.get_forecast(steps=steps)
    mean_forecast = arima_forecast.predicted_mean  # This is the ARIMA mean prediction

    # Forecast the volatility (GARCH part)
    garch_forecast = garch_model.forecast(horizon=steps)
    # garch_forecast.variance is a matrix of forecasted variances
    # shape is typically [time, horizon], so we take the last time step ([-1]) 
    # for each day in the horizon
    vol_forecast_array = garch_forecast.variance.values[-1]  # last row => predicted variance

    # Combine into a DataFrame
    forecast_df = pd.DataFrame({
        'mean_return': mean_forecast,
        'volatility': np.sqrt(vol_forecast_array)
    })

    return forecast_df