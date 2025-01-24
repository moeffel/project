"""
model.py - ARIMA-GARCH with auto-tuning and distribution selection.
"""

import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from typing import Tuple


def fit_arima_garch(
    train_returns: pd.Series,
    arima_order: Tuple[int,int,int] = (1,0,1),
    garch_order: Tuple[int,int] = (1,1),
    dist: str = 'normal',  # 'normal', 't', or 'skewt' in arch
    rescale_data: bool = True,
    scale_factor: float = 1000.0
) -> Tuple[object, object, float]:
    """
    Fits ARIMA + GARCH(arch) on train_returns with user-selected distribution.

    Parameters
    ----------
    train_returns : pd.Series
        Time series (log returns) to model.
    arima_order : tuple
        (p, d, q) for ARIMA.
    garch_order : tuple
        (p, q) for GARCH.
    dist : str
        Distribution for GARCH errors: 'normal', 't', 'skewt', 'ged', etc.
    rescale_data : bool
        Whether to multiply returns by scale_factor for numerical stability.
    scale_factor : float
        If rescale_data=True, multiply each data point by this factor.

    Returns
    -------
    (arima_model, garch_res, used_scale)
    """
    used_scale = 1.0
    if rescale_data:
        used_scale = scale_factor
        train_returns = train_returns * used_scale

    # 1) Fit ARIMA
    try:
        arima_model = ARIMA(train_returns, order=arima_order).fit()
    except ValueError as ve:
        raise ValueError(f"ARIMA fitting failed: {str(ve)}") from ve

    # 2) Fit GARCH with chosen distribution
    try:
        garch = arch_model(
            arima_model.resid,
            p=garch_order[0],
            q=garch_order[1],
            vol='GARCH',
            dist=dist,  # <--- correct arch argument
            rescale=False,
            mean='Zero'  # Typically Zero if ARIMA handles the mean
        )
        garch_res = garch.fit(disp='off')
        if garch_res.convergence_flag != 0:
            raise RuntimeError("GARCH failed to converge.")
    except Exception as e:
        raise RuntimeError(f"GARCH fitting failed: {str(e)}") from e

    return arima_model, garch_res, used_scale


def forecast_arima_garch(
    arima_model,
    garch_model,
    steps: int = 30,
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    Creates a forecast from ARIMA + GARCH.

    Returns a DataFrame with columns:
      - mean_return
      - volatility
    """
    try:
        # 1) ARIMA forecast
        arima_forecast = arima_model.get_forecast(steps=steps)
        mean_return_scaled = arima_forecast.predicted_mean
        # Undo scaling
        mean_return = mean_return_scaled / scale_factor

        # 2) GARCH forecast
        garch_forecast = garch_model.forecast(horizon=steps)
        # arch_model forecast.variance.values[-1].shape = (steps,)
        variance_scaled = garch_forecast.variance.values[-1]
        volatility = np.sqrt(variance_scaled) / scale_factor

        return pd.DataFrame({
            'mean_return': mean_return,
            'volatility': volatility
        })
    except Exception as e:
        raise RuntimeError(f"Forecasting failed: {str(e)}") from e


def auto_tune_arima_garch(series: pd.Series) -> dict:
    """
    Grid search over (p,d,q) for ARIMA and (p,q) for GARCH with 'normal' dist.
    Returns best { 'arima':(p,d,q), 'garch':(p,q) } by minimum AIC.
    """
    best_aic = np.inf
    best_params = {'arima': (1,0,1), 'garch': (1,1)}

    # ARIMA grids
    arima_p = [0, 1, 2, 3]
    arima_d = [0, 1]
    arima_q = [0, 1, 2, 3]
    arima_candidates = list(itertools.product(arima_p, arima_d, arima_q))
    # Remove the trivial (0,*,0) combos
    arima_candidates = [c for c in arima_candidates if not (c[0] == 0 and c[2] == 0)]

    # GARCH grids
    garch_candidates = [(p, q) for p in range(1, 4) for q in range(1, 4)]

    # Search
    for arima_order in arima_candidates:
        try:
            # Fit ARIMA
            # Some older statsmodels versions don't allow disp=0
            arima = ARIMA(series, order=arima_order).fit()
            
            for garch_order in garch_candidates:
                try:
                    # GARCH with normal distribution for speed
                    garch = arch_model(
                        arima.resid,
                        p=garch_order[0],
                        q=garch_order[1],
                        vol='GARCH',
                        dist='normal',
                        mean='Zero'
                    ).fit(disp='off')

                    # Quick check if parameters blow up
                    if np.abs(garch.params).sum() >= 100:
                        continue

                    total_aic = arima.aic + garch.aic
                    if total_aic < best_aic:
                        best_aic = total_aic
                        best_params = {
                            'arima': arima_order,
                            'garch': garch_order
                        }

                except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
                    continue

        except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
            continue

    return best_params
