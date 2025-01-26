# tests/test_utils.py

"""
Test Suite for utils.utils.py

This module contains unit tests for the utilities provided in utils.py, including
descriptive statistics computation, stationarity checks, error metrics, and
residual diagnostics.

To run the tests, navigate to the project directory and execute:
    pytest tests/test_utils.py

Ensure that pytest is installed in your environment:
    pip install pytest
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from utils.utils import (
    compute_descriptive_stats,
    adf_test,
    ljung_box_test,
    arch_test,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    difference_series
)

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def example_df():
    """Fixture for a sample DataFrame with 'price' and 'log_return' columns."""
    data = {
        'price': [100, 105, 102, 110, 108],
        'log_return': [0, 0.05, -0.028, 0.076, -0.018]
    }
    return pd.DataFrame(data)

@pytest.fixture
def residuals():
    """Fixture for a sample residuals array."""
    return [0.1, -0.05, 0.02, 0.01, -0.08, 0.03, -0.02]

@pytest.fixture
def y_true():
    """Fixture for true values."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def y_pred():
    """Fixture for predicted values."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def y_pred_error():
    """Fixture for predicted values with error."""
    return [1.1, 1.9, 3.2, 3.8, 5.1]

# ------------------------------------------------------------------------------
# 1) DESCRIPTIVE STATISTICS TESTS
# ------------------------------------------------------------------------------

def test_compute_descriptive_stats_with_full_data(example_df):
    """
    Test compute_descriptive_stats with a DataFrame containing both 'price' and 'log_return'.
    """
    stats = compute_descriptive_stats(example_df)
    
    assert 'price_mean' in stats
    assert 'logret_mean' in stats
    assert stats['price_mean'] == 105.0
    assert np.isclose(stats['logret_mean'], 0.08, atol=1e-6)

def test_compute_descriptive_stats_with_missing_columns():
    """
    Test compute_descriptive_stats with a DataFrame missing 'log_return'.
    """
    df = pd.DataFrame({'price': [100, 200, 300]})
    stats = compute_descriptive_stats(df)
    
    assert 'price_mean' in stats
    assert 'logret_mean' not in stats
    assert stats['price_mean'] == 200.0

def test_compute_descriptive_stats_empty_df():
    """
    Test compute_descriptive_stats with an empty DataFrame.
    """
    df = pd.DataFrame()
    stats = compute_descriptive_stats(df)
    
    assert stats == {}

# ------------------------------------------------------------------------------
# 2) AUGMENTED DICKEY-FULLER TESTS
# ------------------------------------------------------------------------------

def test_adf_test_stationary():
    """
    Test adf_test with a stationary series.
    """
    # Generate a stationary series (white noise)
    np.random.seed(0)
    series = np.random.normal(0, 1, 100)
    result = adf_test(series)
    
    assert 'test_statistic' in result
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert 'critical_values' in result
    # Since it's white noise, it should be stationary
    assert result['is_stationary'] == True

def test_adf_test_non_stationary():
    """
    Test adf_test with a non-stationary series (random walk).
    """
    np.random.seed(0)
    steps = np.random.normal(0, 1, 100)
    series = np.cumsum(steps)  # Random walk
    result = adf_test(series)
    
    assert 'is_stationary' in result
    # Random walk is non-stationary
    assert result['is_stationary'] == False

def test_adf_test_with_small_series():
    """
    Test adf_test with a very short series.
    """
    series = [1, 2]
    with pytest.raises(ValueError, match="sample size is too short"):
        adf_test(series)

# ------------------------------------------------------------------------------
# 3) RESIDUAL ANALYSIS & DIAGNOSTICS TESTS
# ------------------------------------------------------------------------------

def test_ljung_box_test_white_noise(residuals):
    """
    Test ljung_box_test with residuals that resemble white noise.
    """
    result = ljung_box_test(residuals, lags=6)
    assert 'lb_stat' in result
    assert 'lb_pvalue' in result
    assert 'is_white_noise' in result
    # With small sample, p-value might not be significant
    assert isinstance(result['is_white_noise'], bool)

def test_ljung_box_test_non_white_noise():
    """
    Test ljung_box_test with residuals that are autocorrelated.
    """
    # Create residuals with autocorrelation
    residuals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = ljung_box_test(residuals, lags=4)
    
    assert result['is_white_noise'] == False

def test_arch_test_no_heteroskedasticity(residuals):
    """
    Test arch_test with residuals that have constant variance.
    """
    result = arch_test(residuals, lags=3)
    assert 'arch_stat' in result
    assert 'arch_pvalue' in result
    assert 'heteroskedastic' in result
    assert result['heteroskedastic'] == False

def test_arch_test_with_heteroskedasticity():
    """
    Test arch_test with residuals that exhibit heteroskedasticity.
    """
    # Create residuals with increasing variance
    residuals = [1, 4, 1, 9, 1, 16, 1, 25]  # Variance increases quadratically
    result = arch_test(residuals, lags=2)
    
    assert 'heteroskedastic' in result
    assert result['heteroskedastic'] == True

# ------------------------------------------------------------------------------
# 4) ERROR METRICS TESTS
# ------------------------------------------------------------------------------

def test_mean_absolute_error(y_true, y_pred):
    """
    Test mean_absolute_error with identical true and predicted values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 0.0

def test_mean_absolute_error_with_errors(y_true, y_pred_error):
    """
    Test mean_absolute_error with some prediction errors.
    """
    mae = mean_absolute_error(y_true, y_pred_error)
    expected_mae = (0.1 + 0.1 + 0.2 + 0.2 + 0.1) / 5
    assert np.isclose(mae, expected_mae, atol=1e-6)

def test_mean_squared_error(y_true, y_pred):
    """
    Test mean_squared_error with identical true and predicted values.
    """
    mse = mean_squared_error(y_true, y_pred)
    assert mse == 0.0

def test_mean_squared_error_with_errors(y_true, y_pred_error):
    """
    Test mean_squared_error with some prediction errors.
    """
    mse = mean_squared_error(y_true, y_pred_error)
    expected_mse = (0.1**2 + 0.1**2 + 0.2**2 + 0.2**2 + 0.1**2) / 5
    assert np.isclose(mse, expected_mse, atol=1e-6)

def test_root_mean_squared_error(y_true, y_pred):
    """
    Test root_mean_squared_error with identical true and predicted values.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse == 0.0

def test_root_mean_squared_error_with_errors(y_true, y_pred_error):
    """
    Test root_mean_squared_error with some prediction errors.
    """
    rmse = root_mean_squared_error(y_true, y_pred_error)
    expected_rmse = np.sqrt(
        (0.1**2 + 0.1**2 + 0.2**2 + 0.2**2 + 0.1**2) / 5
    )
    assert np.isclose(rmse, expected_rmse, atol=1e-6)

def test_mean_absolute_percentage_error(y_true, y_pred):
    """
    Test mean_absolute_percentage_error with identical true and predicted values.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert mape == 0.0

def test_mean_absolute_percentage_error_with_errors(y_true, y_pred_error):
    """
    Test mean_absolute_percentage_error with some prediction errors.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred_error)
    # ((0.1/1) + (0.1/2) + (0.2/3) + (0.2/4) + (0.1/5)) / 5
    expected_mape = (0.1 + 0.05 + 0.0666666667 + 0.05 + 0.02) / 5
    assert np.isclose(mape, expected_mape, atol=1e-6)

def test_error_metrics_with_empty_inputs():
    """
    Test error metrics functions with empty input lists.
    """
    empty_true = []
    empty_pred = []
    mae = mean_absolute_error(empty_true, empty_pred)
    mse = mean_squared_error(empty_true, empty_pred)
    rmse = root_mean_squared_error(empty_true, empty_pred)
    
    # Since the inputs are empty, the mean operations result in nan
    assert np.isnan(mae)
    assert np.isnan(mse)
    assert np.isnan(rmse)
    
    # mean_absolute_percentage_error should raise ValueError due to division by zero
    with pytest.raises(ValueError, match="y_true contains zero"):
        mean_absolute_percentage_error(empty_true, empty_pred)

# ------------------------------------------------------------------------------
# 5) DIFFERENCING TESTS
# ------------------------------------------------------------------------------

def test_difference_series_first_order():
    """
    Test difference_series with order=1.
    """
    series = pd.Series([1, 2, 3, 4, 5])
    diff = difference_series(series, order=1)
    expected = pd.Series([1.0, 1.0, 1.0, 1.0], dtype=float).reset_index(drop=True)
    pd.testing.assert_series_equal(diff.reset_index(drop=True), expected)

def test_difference_series_second_order():
    """
    Test difference_series with order=2.
    """
    series = pd.Series([1, 2, 3, 4, 5])
    diff = difference_series(series, order=2)
    expected = pd.Series([0.0, 0.0, 0.0], dtype=float).reset_index(drop=True)
    pd.testing.assert_series_equal(diff.reset_index(drop=True), expected)

def test_difference_series_order_zero():
    """
    Test difference_series with order=0, which should return the original series.
    """
    series = pd.Series([1, 2, 3, 4, 5])
    diff = difference_series(series, order=0)
    expected = series.copy().astype(float).reset_index(drop=True)
    pd.testing.assert_series_equal(diff.reset_index(drop=True), expected)

def test_difference_series_order_exceeds_length():
    """
    Test difference_series with order exceeding the series length.
    """
    series = pd.Series([1, 2])
    diff = difference_series(series, order=3)
    expected = pd.Series(dtype=float)
    pd.testing.assert_series_equal(diff, expected)

def test_difference_series_with_missing_values():
    """
    Test difference_series with a series containing NaN values.
    """
    series = pd.Series([1, 2, np.nan, 4, 5])
    diff = difference_series(series, order=1)
    expected = pd.Series([1.0, np.nan, 4.0, 1.0], dtype=float).reset_index(drop=True)
    pd.testing.assert_series_equal(diff.reset_index(drop=True), expected)

# ------------------------------------------------------------------------------
# 6) DOCTESTS FOR UTILITIES
# ------------------------------------------------------------------------------

def test_doctests():
    """
    Run doctests for utils.py.

    This test ensures that all doctests within utils.py pass successfully.
    """
    import doctest
    import utils.utils
    doctest.testmod(utils.utils)

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main()