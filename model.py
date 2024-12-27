"""
model.py

Implements ARIMA+GARCH with optional data rescaling to avoid arch DataScaleWarning.
Returns 3 values: (arima_model, garch_res, scale_factor).

Author: Your Name
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from typing import Tuple, Optional

def fit_arima_garch(
    train_returns: pd.Series,
    arima_order: Tuple[int,int,int] = (1,0,1),
    garch_order: Tuple[int,int] = (1,1),
    dist: str = 'normal',
    rescale_data: bool = True,
    scale_factor: float = 1000.0
):
    """
    Fits an ARIMA model, then a GARCH model on ARIMA residuals,
    optionally rescaling data to avoid the arch DataScaleWarning.

    Parameters
    ----------
    train_returns : pd.Series
        Log returns for training (original scale).
    arima_order : (p, d, q)
        Default (1,0,1).
    garch_order : (p, q)
        Default (1,1).
    dist : str
        Distribution for GARCH: 'normal', 't', or 'skewt'.
    rescale_data : bool
        If True, multiply train_returns by scale_factor prior to modeling.
    scale_factor : float
        How much to multiply the returns by. E.g. 1000 or 10000.

    Returns
    -------
    (ARIMAResults, ARCHModelResult, float)
        - arima_model : statsmodels ARIMA fit
        - garch_res   : arch GARCH fit
        - used_scale  : the actual scale factor used (1.0 if rescale_data=False)

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([0.001, 0.002, -0.001, 0.0005])
    >>> arima_model, garch_res, scale_factor = fit_arima_garch(data, (1,0,1), (1,1), dist='normal')
    """
    # Decide whether to scale
    used_scale = 1.0
    if rescale_data:
        used_scale = scale_factor
        train_returns = train_returns * used_scale

    # 1) Fit ARIMA
    arima_model = ARIMA(train_returns, order=arima_order).fit()
    resid = arima_model.resid

    # 2) Fit GARCH
    #    We pass rescale=False so arch won't attempt automatic rescaling.
    garch = arch_model(resid, p=garch_order[0], q=garch_order[1], dist=dist, rescale=False)
    garch_res = garch.fit(disp="off")

    return arima_model, garch_res, used_scale

def forecast_arima_garch(
    arima_model,
    garch_model,
    steps: int = 30,
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    Forecasts future log returns using ARIMA + GARCH,
    then unscales them if they were scaled.

    Parameters
    ----------
    arima_model : ARIMAResults
    garch_model : ARCHModelResult
    steps : int
        Number of forecast days.
    scale_factor : float
        If we scaled data by e.g. 1000, we must unscale the predictions by /1000.

    Returns
    -------
    pd.DataFrame
        Columns: ['mean_return', 'volatility']

    Examples
    --------
    (see fit_arima_garch docstring for usage)
    """
    # 1) ARIMA forecast (in scaled space)
    arima_forecast = arima_model.get_forecast(steps=steps)
    mean_forecast_scaled = arima_forecast.predicted_mean

    # 2) GARCH forecast (in scaled space)
    garch_forecast = garch_model.forecast(horizon=steps)
    vol_forecast_array = garch_forecast.variance.values[-1]
    volatility_scaled = np.sqrt(vol_forecast_array)

    # 3) Unscale if needed
    if scale_factor != 1.0:
        mean_return = mean_forecast_scaled / scale_factor
        volatility = volatility_scaled / scale_factor
    else:
        mean_return = mean_forecast_scaled
        volatility = volatility_scaled

    return pd.DataFrame({
        'mean_return': mean_return,
        'volatility': volatility
    })


if __name__ == "__main__":
    # Minimal quick test
    data = pd.Series([0.001, 0.002, -0.001, 0.0005, 0.002])
    arima_model, garch_res, used_scale = fit_arima_garch(
        data,
        arima_order=(1,0,1),
        garch_order=(1,1),
        dist='normal',
        rescale_data=True,
        scale_factor=1000
    )
    fc = forecast_arima_garch(arima_model, garch_res, steps=5, scale_factor=used_scale)
    print("Forecast DataFrame:\n", fc)