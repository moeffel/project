"""
test_model.py - Pytest Suite for ARIMA-GARCH Model

This test suite verifies the functionality of the ARIMA-GARCH modeling functions
defined in model.py. It includes tests for fitting the model, forecasting, and
auto-tuning of hyperparameters.

Usage:
    To run the tests, navigate to the project directory and execute:
        pytest test_model.py
"""

import pytest
import pandas as pd
import numpy as np
from model import fit_arima_garch, forecast_arima_garch, auto_tune_arima_garch
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

@pytest.fixture
def sample_series():
    """
    Fixture to create a sample time series data for testing.

    Returns:
        pd.Series: Simulated log returns with ARIMA(1,0,1) properties.
    """
    np.random.seed(42)
    # Simulate ARIMA(1,0,1) data
    ar = np.array([0.5])
    ma = np.array([0.5])
    n = 1000
    from statsmodels.tsa.arima_process import ArmaProcess
    arma_process = ArmaProcess(ar, ma)
    simulated_data = arma_process.generate_sample(nsample=n)
    return pd.Series(simulated_data)

@pytest.fixture
def trained_models(sample_series):
    """
    Fixture to fit ARIMA-GARCH models on the sample series.

    Returns:
        tuple: (arima_model, garch_res, used_scale)
    """
    arima_order = (1, 0, 1)
    garch_order = (1, 1)
    dist = 'normal'
    return fit_arima_garch(
        train_returns=sample_series,
        arima_order=arima_order,
        garch_order=garch_order,
        dist=dist,
        rescale_data=True,
        scale_factor=1000.0
    )

def test_fit_arima_garch(trained_models):
    """
    Test the fit_arima_garch function for successful model fitting.

    Args:
        trained_models (tuple): Tuple containing fitted ARIMA and GARCH models and the scale used.
    """
    arima_model, garch_res, used_scale = trained_models

    # Check that the returned models are instances of their respective classes
    assert isinstance(arima_model, ARIMA), "arima_model is not an instance of ARIMA."
    assert isinstance(garch_res, arch_model), "garch_res is not an instance of arch_model."

    # Check that used_scale is correctly returned
    assert used_scale == 1000.0, "Scale factor is not correctly returned."

    # Verify that ARIMA model has been fitted by checking summary
    assert arima_model.summary(), "ARIMA model summary is empty."

    # Verify that GARCH model has been fitted by checking summary
    assert garch_res.summary(), "GARCH model summary is empty."

def test_forecast_arima_garch(trained_models):
    """
    Test the forecast_arima_garch function for generating forecasts.

    Args:
        trained_models (tuple): Tuple containing fitted ARIMA and GARCH models and the scale used.
    """
    arima_model, garch_res, used_scale = trained_models

    steps = 10
    forecast_df = forecast_arima_garch(
        arima_model=arima_model,
        garch_model=garch_res,
        steps=steps,
        scale_factor=1000.0
    )

    # Check that the forecast is a DataFrame
    assert isinstance(forecast_df, pd.DataFrame), "Forecast is not a pandas DataFrame."

    # Check that the DataFrame has the correct columns
    expected_columns = {'mean_return', 'volatility'}
    assert expected_columns.issubset(forecast_df.columns), "Forecast DataFrame missing expected columns."

    # Check that the number of forecast steps is correct
    assert len(forecast_df) == steps, f"Forecast DataFrame does not have {steps} steps."

    # Check that mean_return and volatility are numeric
    assert pd.api.types.is_numeric_dtype(forecast_df['mean_return']), "mean_return is not numeric."
    assert pd.api.types.is_numeric_dtype(forecast_df['volatility']), "volatility is not numeric."

def test_auto_tune_arima_garch(sample_series):
    """
    Test the auto_tune_arima_garch function for parameter selection.

    Args:
        sample_series (pd.Series): Sample time series data.
    """
    best_params = auto_tune_arima_garch(series=sample_series)

    # Check that the best_params is a dictionary with required keys
    assert isinstance(best_params, dict), "Best parameters is not a dictionary."
    assert 'arima' in best_params, "Best parameters missing 'arima' key."
    assert 'garch' in best_params, "Best parameters missing 'garch' key."

    # Check that ARIMA order is a tuple of three integers
    assert isinstance(best_params['arima'], tuple), "ARIMA order is not a tuple."
    assert len(best_params['arima']) == 3, "ARIMA order does not have three elements."
    assert all(isinstance(x, int) for x in best_params['arima']), "ARIMA order elements are not all integers."

    # Check that GARCH order is a tuple of two integers
    assert isinstance(best_params['garch'], tuple), "GARCH order is not a tuple."
    assert len(best_params['garch']) == 2, "GARCH order does not have two elements."
    assert all(isinstance(x, int) for x in best_params['garch']), "GARCH order elements are not all integers."

def test_fit_arima_garch_invalid_order(sample_series):
    """
    Test fit_arima_garch with invalid ARIMA order to ensure it raises ValueError.

    Args:
        sample_series (pd.Series): Sample time series data.
    """
    invalid_arima_order = (5, 2, 5)  # Likely invalid for the sample data

    with pytest.raises(ValueError, match="ARIMA fitting failed"):
        fit_arima_garch(
            train_returns=sample_series,
            arima_order=invalid_arima_order,
            garch_order=(1, 1),
            dist='normal',
            rescale_data=True,
            scale_factor=1000.0
        )

def test_forecast_arima_garch_invalid_model(trained_models):
    """
    Test forecast_arima_garch with invalid models to ensure it raises RuntimeError.

    Args:
        trained_models (tuple): Tuple containing fitted ARIMA and GARCH models and the scale used.
    """
    arima_model, garch_res, used_scale = trained_models

    # Intentionally pass an incorrect GARCH model
    invalid_garch_model = None

    with pytest.raises(RuntimeError, match="Forecasting failed"):
        forecast_arima_garch(
            arima_model=arima_model,
            garch_model=invalid_garch_model,
            steps=10,
            scale_factor=1000.0
        )

def test_auto_tune_arima_garch_output(sample_series):
    """
    Additional test to verify that auto_tune_arima_garch returns reasonable parameters.

    Args:
        sample_series (pd.Series): Sample time series data.
    """
    best_params = auto_tune_arima_garch(series=sample_series)

    # Check that the ARIMA parameters are within the specified grid
    arima_p, arima_d, arima_q = best_params['arima']
    assert 0 <= arima_p <= 3, "ARIMA p parameter out of grid range."
    assert arima_d in [0, 1], "ARIMA d parameter out of grid range."
    assert 0 <= arima_q <= 3, "ARIMA q parameter out of grid range."

    # Check that the GARCH parameters are within the specified grid
    garch_p, garch_q = best_params['garch']
    assert 1 <= garch_p <= 3, "GARCH p parameter out of grid range."
    assert 1 <= garch_q <= 3, "GARCH q parameter out of grid range."

@pytest.mark.parametrize("dist", ['normal', 't', 'skewt'])
def test_fit_arima_garch_various_distributions(sample_series, dist):
    """
    Test fit_arima_garch with different GARCH error distributions.

    Args:
        sample_series (pd.Series): Sample time series data.
        dist (str): Distribution to test ('normal', 't', 'skewt').
    """
    arima_order = (1, 0, 1)
    garch_order = (1, 1)

    try:
        arima_model, garch_res, used_scale = fit_arima_garch(
            train_returns=sample_series,
            arima_order=arima_order,
            garch_order=garch_order,
            dist=dist,
            rescale_data=True,
            scale_factor=1000.0
        )

        # Check that the GARCH model uses the correct distribution
        assert garch_res.distribution.name == dist, f"GARCH distribution is not {dist}."

    except Exception as e:
        pytest.fail(f"fit_arima_garch failed with distribution '{dist}': {e}")

def test_auto_tune_arima_garch_empty_series():
    """
    Test auto_tune_arima_garch with an empty series to ensure it handles gracefully.
    """
    empty_series = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        auto_tune_arima_garch(series=empty_series)

def test_fit_arima_garch_rescale_false(sample_series):
    """
    Test fit_arima_garch with rescale_data=False to ensure scaling is handled correctly.

    Args:
        sample_series (pd.Series): Sample time series data.
    """
    arima_order = (1, 0, 1)
    garch_order = (1, 1)
    scale_factor = 1000.0

    arima_model, garch_res, used_scale = fit_arima_garch(
        train_returns=sample_series,
        arima_order=arima_order,
        garch_order=garch_order,
        dist='normal',
        rescale_data=False,
        scale_factor=scale_factor
    )

    # Check that used_scale is 1.0 when rescale_data is False
    assert used_scale == 1.0, "used_scale should be 1.0 when rescale_data is False."

def test_forecast_arima_garch_output_structure(trained_models):
    """
    Verify that the forecast_arima_garch output has no NaN values and realistic values.

    Args:
        trained_models (tuple): Tuple containing fitted ARIMA and GARCH models and the scale used.
    """
    arima_model, garch_res, used_scale = trained_models
    steps = 10
    forecast_df = forecast_arima_garch(
        arima_model=arima_model,
        garch_model=garch_res,
        steps=steps,
        scale_factor=1000.0
    )

    # Ensure there are no NaN values
    assert not forecast_df.isnull().values.any(), "Forecast contains NaN values."

    # Check that mean_return and volatility are within reasonable bounds
    assert forecast_df['mean_return'].between(-1, 1).all(), "mean_return values are out of expected range."
    assert forecast_df['volatility'].between(0, 5).all(), "volatility values are out of expected range."