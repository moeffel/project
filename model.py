"""
model.py - Updated with enhanced auto-tuning
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
    dist: str = 'normal',
    rescale_data: bool = True,
    scale_factor: float = 1000.0
) -> Tuple[object, object, float]:
    """
    Returns 3 values: (arima_model, garch_res, scale_factor)
    """
    used_scale = 1.0
    if rescale_data:
        used_scale = scale_factor
        train_returns = train_returns * used_scale

    # Fit ARIMA with improved error handling
    try:
        arima_model = ARIMA(train_returns, order=arima_order).fit()
    except ValueError as ve:
        raise ValueError(f"ARIMA fitting failed: {str(ve)}") from ve
        
    # Fit GARCH with convergence checks
    try:
        garch = arch_model(arima_model.resid, p=garch_order[0], q=garch_order[1], 
                          dist=dist, rescale=False)
        garch_res = garch.fit(disp='off')
        if not garch_res.convergence_flag == 0:
            raise RuntimeError("GARCH failed to converge")
    except Exception as e:
        raise RuntimeError(f"GARCH fitting failed: {str(e)}") from e

    return arima_model, garch_res, used_scale

def forecast_arima_garch(
    arima_model,
    garch_model,
    steps: int = 30,
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """Forecast with enhanced stability checks"""
    try:
        # ARIMA forecast
        arima_forecast = arima_model.get_forecast(steps=steps)
        mean_return = arima_forecast.predicted_mean / scale_factor
        
        # GARCH forecast
        garch_forecast = garch_model.forecast(horizon=steps)
        volatility = np.sqrt(garch_forecast.variance.values[-1]) / scale_factor

        return pd.DataFrame({
            'mean_return': mean_return,
            'volatility': volatility
        })
    except Exception as e:
        raise RuntimeError(f"Forecasting failed: {str(e)}") from e

def auto_tune_arima_garch(series):
    """Enhanced auto-tuning with expanded grid search"""
    best_aic = np.inf
    best_params = {'arima': (1,0,1), 'garch': (1,1)}
    
    # Expanded ARIMA grid
    arima_p = [0, 1, 2, 3]
    arima_d = [0, 1]
    arima_q = [0, 1, 2, 3]
    arima_candidates = list(itertools.product(arima_p, arima_d, arima_q))
    arima_candidates = [c for c in arima_candidates if not (c[0] == 0 and c[2] == 0)]
    
    # Expanded GARCH grid
    garch_candidates = [(p, q) for p in range(1, 4) for q in range(1, 4)]
    
    # Grid search with stability checks
    for arima_order in arima_candidates:
        try:
            # Fit ARIMA with silent mode
            arima = ARIMA(series, order=arima_order).fit(disp=0)
            
            for garch_order in garch_candidates:
                try:
                    # Fit GARCH with normal errors
                    garch = arch_model(
                        arima.resid, 
                        p=garch_order[0], 
                        q=garch_order[1], 
                        dist='normal'
                    ).fit(disp='off', show_warning=False)
                    
                    # Stability check
                    if np.abs(garch.params).sum() >= 100:
                        continue
                        
                    # AIC comparison
                    total_aic = arima.aic + garch.aic
                    if total_aic < best_aic:
                        best_aic = total_aic
                        best_params = {
                            'arima': arima_order,
                            'garch': garch_order
                        }
                except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
                    continue
        except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
            continue
    
    return best_params
