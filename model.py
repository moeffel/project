"""
model.py - ARIMA-GARCH with Auto-Tuning and Distribution Selection

This module provides functions to fit ARIMA and GARCH models to financial time series data,
perform forecasting, and automatically tune model parameters using grid search. It leverages
the `statsmodels` and `arch` libraries for time series modeling.

Functions:
- fit_arima_garch: Fit ARIMA and GARCH models to the provided return series.
- forecast_arima_garch: Generate forecasts using the fitted ARIMA and GARCH models.
- auto_tune_arima_garch: Automatically select the best ARIMA and GARCH orders based on AIC.

Example:
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.randn(1000) / 100)
    >>> best_params = auto_tune_arima_garch(returns)
    >>> best_params
    {'arima': (1, 0, 1), 'garch': (1, 1)}
"""

import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from typing import Tuple


def fit_arima_garch(
    train_returns: pd.Series,
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    garch_order: Tuple[int, int] = (1, 1),
    dist: str = 'normal',  # 'normal', 't', or 'skewt' in arch
    rescale_data: bool = True,
    scale_factor: float = 1000.0
) -> Tuple[object, object, float]:
    """
    Fit ARIMA and GARCH models to the training returns with specified orders and distribution.

    This function first optionally rescales the input return series for numerical stability,
    fits an ARIMA model to capture the mean dynamics, and then fits a GARCH model to the
    residuals from the ARIMA model to capture volatility clustering.

    Parameters
    ----------
    train_returns : pd.Series
        Time series of log returns to model.
    arima_order : tuple of int, default (1, 0, 1)
        The (p, d, q) order of the ARIMA model.
    garch_order : tuple of int, default (1, 1)
        The (p, q) order of the GARCH model.
    dist : str, default 'normal'
        The distribution to use for GARCH errors. Options include 'normal', 't', 'skewt', 'ged', etc.
    rescale_data : bool, default True
        Whether to multiply returns by `scale_factor` for numerical stability during modeling.
    scale_factor : float, default 1000.0
        The factor by which to scale the data if `rescale_data` is True.

    Returns
    -------
    tuple
        A tuple containing:
        - arima_model: The fitted ARIMA model object.
        - garch_res: The fitted GARCH model results object.
        - used_scale: The scale factor used (1.0 if not rescaled).

    Raises
    ------
    ValueError
        If ARIMA fitting fails due to invalid parameters or data issues.
    RuntimeError
        If GARCH fitting fails to converge or encounters other issues.

    Example
    -------
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.randn(1000) / 100)
    >>> arima_model, garch_res, scale = fit_arima_garch(returns)
    >>> isinstance(arima_model, ARIMA)
    True
    >>> 'GARCH' in str(garch_res)
    True
    >>> scale
    1000.0
    """
    used_scale = 1.0  # Default scale factor
    if rescale_data:
        used_scale = scale_factor
        train_returns = train_returns * used_scale
        # Debug: Print scaling information
        # print(f"Data rescaled by factor {scale_factor}")

    # 1) Fit ARIMA Model
    try:
        arima_model = ARIMA(train_returns, order=arima_order).fit()
        # Debug: Print ARIMA summary
        # print(arima_model.summary())
    except ValueError as ve:
        raise ValueError(f"ARIMA fitting failed: {str(ve)}") from ve
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during ARIMA fitting: {str(e)}") from e

    # 2) Fit GARCH Model to ARIMA Residuals
    try:
        garch = arch_model(
            arima_model.resid,
            p=garch_order[0],
            q=garch_order[1],
            vol='GARCH',
            dist=dist,
            rescale=False,  # Data already rescaled if needed
            mean='Zero'  # Assume ARIMA has captured the mean
        )
        garch_res = garch.fit(disp='off')  # Suppress output
        # Check for convergence
        if garch_res.convergence_flag != 0:
            raise RuntimeError("GARCH failed to converge.")
        # Debug: Print GARCH summary
        # print(garch_res.summary())
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
    Generate forecasts for mean returns and volatility using fitted ARIMA and GARCH models.

    This function forecasts future mean returns using the ARIMA model and future volatility
    using the GARCH model. The forecasts are returned in a pandas DataFrame.

    Parameters
    ----------
    arima_model : object
        The fitted ARIMA model object.
    garch_model : object
        The fitted GARCH model results object.
    steps : int, default 30
        The number of future time steps to forecast.
    scale_factor : float, default 1.0
        The scale factor used during model fitting. Used to rescale forecasts if data was scaled.

    Returns
    -------
    pd.DataFrame
        A DataFrame with `steps` rows and two columns:
        - 'mean_return': Forecasted mean returns.
        - 'volatility': Forecasted volatility (standard deviation).

    Raises
    ------
    RuntimeError
        If forecasting fails due to issues in the ARIMA or GARCH models.

    Example
    -------
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.randn(1000) / 100)
    >>> arima_model, garch_res, scale = fit_arima_garch(returns)
    >>> forecast = forecast_arima_garch(arima_model, garch_res, steps=5, scale_factor=scale)
    >>> forecast.shape
    (5, 2)
    >>> list(forecast.columns)
    ['mean_return', 'volatility']
    """
    try:
        # 1) Forecast Mean Returns using ARIMA
        arima_forecast = arima_model.get_forecast(steps=steps)
        mean_return_scaled = arima_forecast.predicted_mean
        # Rescale mean returns if data was scaled
        mean_return = mean_return_scaled / scale_factor

        # 2) Forecast Volatility using GARCH
        garch_forecast = garch_model.forecast(horizon=steps)
        # Extract the forecasted variances from the last observation
        variance_scaled = garch_forecast.variance.values[-1]
        # Compute standard deviation (volatility) and rescale
        volatility = np.sqrt(variance_scaled) / scale_factor

        # Construct the forecast DataFrame
        forecast_df = pd.DataFrame({
            'mean_return': mean_return,
            'volatility': volatility
        })

        return forecast_df

    except Exception as e:
        raise RuntimeError(f"Forecasting failed: {str(e)}") from e


def auto_tune_arima_garch(series: pd.Series) -> dict:
    """
    Automatically tune ARIMA and GARCH model orders using grid search based on AIC.

    This function performs an exhaustive search over specified ranges of ARIMA (p, d, q)
    and GARCH (p, q) orders. For each combination, it fits the models and selects the
    combination with the lowest total Akaike Information Criterion (AIC).

    Parameters
    ----------
    series : pd.Series
        Time series of log returns to model.

    Returns
    -------
    dict
        A dictionary containing the best ARIMA and GARCH orders:
        {
            'arima': (p, d, q),
            'garch': (p, q)
        }

    Example
    -------
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.randn(1000) / 100)
    >>> best_params = auto_tune_arima_garch(returns)
    >>> isinstance(best_params, dict)
    True
    >>> 'arima' in best_params and 'garch' in best_params
    True
    """
    best_aic = np.inf  # Initialize best AIC to infinity
    best_params = {'arima': (1, 0, 1), 'garch': (1, 1)}  # Default parameters

    # Define ARIMA parameter grid
    arima_p = [0, 1, 2, 3]
    arima_d = [0, 1]
    arima_q = [0, 1, 2, 3]
    arima_candidates = list(itertools.product(arima_p, arima_d, arima_q))
    # Exclude trivial ARIMA models where both p and q are zero
    arima_candidates = [c for c in arima_candidates if not (c[0] == 0 and c[2] == 0)]

    # Define GARCH parameter grid
    garch_candidates = list(itertools.product(range(1, 4), range(1, 4)))  # (1,1) to (3,3)

    # Iterate over all ARIMA and GARCH combinations
    for arima_order in arima_candidates:
        try:
            # Fit ARIMA model
            arima = ARIMA(series, order=arima_order).fit()
        except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
            # Skip invalid ARIMA configurations
            continue
        except Exception:
            # Catch-all for unexpected exceptions
            continue

        for garch_order in garch_candidates:
            try:
                # Fit GARCH model with normal distribution for speed
                garch = arch_model(
                    arima.resid,
                    p=garch_order[0],
                    q=garch_order[1],
                    vol='GARCH',
                    dist='normal',
                    mean='Zero'
                ).fit(disp='off')

                # Check for unrealistic parameter values
                if np.abs(garch.params).sum() >= 100:
                    continue  # Skip models with parameters that have exploded

                # Calculate total AIC as the sum of ARIMA and GARCH AICs
                total_aic = arima.aic + garch.aic

                # Update best parameters if current AIC is lower
                if total_aic < best_aic:
                    best_aic = total_aic
                    best_params = {
                        'arima': arima_order,
                        'garch': garch_order
                    }

            except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
                # Skip invalid GARCH configurations
                continue
            except Exception:
                # Catch-all for unexpected exceptions
                continue

    return best_params