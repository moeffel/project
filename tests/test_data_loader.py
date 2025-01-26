# test_data_loader.py

"""
Pytest Suite for data_loader.py

This test suite covers the functionality provided by the data_loader.py module,
ensuring reliable data retrieval, preprocessing, splitting, and statistical analysis
of cryptocurrency price data fetched from Yahoo Finance.

Usage:
    1. Ensure pytest is installed:
        pip install pytest

    2. Run the tests:
        pytest test_data_loader.py

    All tests should pass, indicating that the data_loader functions are working as expected.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime
from data_loader import (
    fetch_data_yahoo,
    preprocess_data,
    train_test_split,
    compute_descriptive_stats,
    fetch_data_for_coins,
    CRYPTO_SYMBOLS
)

# ------------------------------------------------------------------------------
# Helper Functions and Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_yahoo_data():
    """
    Fixture to provide sample data mimicking yfinance's downloaded DataFrame.
    """
    data = {
        'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
        'Close': [100, 105, 110, 115, 120]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_yahoo_data_multilevel():
    """
    Fixture to provide sample data with multi-level columns as returned by yfinance.
    """
    arrays = [
        ['Close', 'Close', 'Volume'],
        ['BTC-USD', 'ETH-USD', 'BTC-USD']
    ]
    tuples = list(zip(*arrays))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['Attribute', 'Symbol'])

    data = [
        [100, 200, 1000],
        [105, 210, 1100],
        [110, 220, 1200],
        [115, 230, 1300],
        [120, 240, 1400],
    ]

    return pd.DataFrame(data, columns=multi_index, index=pd.date_range(start='2022-01-01', periods=5, freq='D'))

# ------------------------------------------------------------------------------
# Tests for fetch_data_yahoo
# ------------------------------------------------------------------------------

@patch('data_loader.yf.download')
def test_fetch_data_yahoo_valid_coin(mock_download, sample_yahoo_data):
    """
    Test fetch_data_yahoo with a valid coin_id and ensure correct data is returned.
    """
    # Mock yfinance download to return sample_yahoo_data
    mock_download.return_value = sample_yahoo_data

    # Call the function with a valid coin_id
    df = fetch_data_yahoo("bitcoin", start="2022-01-01", end="2022-01-05")

    # Assertions
    assert not df.empty, "DataFrame should not be empty"
    assert list(df.columns) == ["date", "price"], "Columns should be ['date', 'price']"
    assert len(df) == 5, "Should have 5 rows of data"
    assert df['price'].iloc[0] == 100, "First price should be 100"
    assert pd.to_datetime(df['date'].iloc[0]) == pd.to_datetime("2022-01-01"), "First date should be 2022-01-01"

@patch('data_loader.yf.download')
def test_fetch_data_yahoo_invalid_coin(mock_download):
    """
    Test fetch_data_yahoo with an invalid coin_id and expect a ValueError.
    """
    # Attempt to call the function with an invalid coin_id
    with pytest.raises(ValueError, match="Unknown coin_id 'invalidcoin'"):
        fetch_data_yahoo("invalidcoin", start="2022-01-01", end="2022-01-05")

@patch('data_loader.yf.download')
def test_fetch_data_yahoo_no_close_column(mock_download, sample_yahoo_data):
    """
    Test fetch_data_yahoo when the downloaded data lacks the 'Close' column.
    Expect a KeyError.
    """
    # Remove 'Close' column to simulate missing data
    mock_data = sample_yahoo_data.drop(columns=['Close'])
    mock_download.return_value = mock_data

    with pytest.raises(KeyError, match="No 'Close' column found"):
        fetch_data_yahoo("bitcoin", start="2022-01-01", end="2022-01-05")

@patch('data_loader.yf.download')
def test_fetch_data_yahoo_empty_data(mock_download):
    """
    Test fetch_data_yahoo when yfinance returns empty data. Expect a ValueError.
    """
    # Mock yfinance to return empty DataFrame
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No data returned for BTC-USD"):
        fetch_data_yahoo("bitcoin", start="2022-01-01", end="2022-01-05")

@patch('data_loader.yf.download')
def test_fetch_data_yahoo_multilevel_columns(mock_download, sample_yahoo_data_multilevel):
    """
    Test fetch_data_yahoo with multi-level columns and ensure columns are flattened correctly.
    """
    # Mock yfinance download to return multi-level columns DataFrame
    mock_download.return_value = sample_yahoo_data_multilevel

    # Call the function with a valid coin_id
    df = fetch_data_yahoo("bitcoin", start="2022-01-01", end="2022-01-05")

    # Assertions
    assert not df.empty, "DataFrame should not be empty"
    assert list(df.columns) == ["date", "price"], "Columns should be ['date', 'price']"
    assert df['price'].iloc[0] == 100, "First price should be 100"
    assert pd.to_datetime(df['date'].iloc[0]) == pd.to_datetime("2022-01-01"), "First date should be 2022-01-01"

# ------------------------------------------------------------------------------
# Tests for preprocess_data
# ------------------------------------------------------------------------------

def test_preprocess_data_normal():
    """
    Test preprocess_data with normal data and ensure correct processing.
    """
    # Create a sample raw DataFrame
    raw_df = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']),
        'price': [100, 105, 110]
    })

    # Call the preprocessing function
    processed_df = preprocess_data(raw_df)

    # Assertions
    assert not processed_df.empty, "Processed DataFrame should not be empty"
    assert 'log_return' in processed_df.columns, "'log_return' column should be present"
    assert len(processed_df) == 2, "After preprocessing, should have 2 rows"
    expected_log_return = np.log(105 / 100)
    np.testing.assert_almost_equal(processed_df['log_return'].iloc[0], expected_log_return, decimal=5)

def test_preprocess_data_with_missing_prices():
    """
    Test preprocess_data with missing price values and ensure interpolation works.
    """
    # Create a sample raw DataFrame with missing price
    raw_df = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']),
        'price': [100, np.nan, 110]
    })

    # Call the preprocessing function
    processed_df = preprocess_data(raw_df)

    # Assertions
    assert not processed_df.empty, "Processed DataFrame should not be empty"
    assert 'log_return' in processed_df.columns, "'log_return' column should be present"
    assert len(processed_df) == 2, "After preprocessing, should have 2 rows"

    # Use pytest.approx for floating point comparison
    assert processed_df['price'].iloc[1] == pytest.approx(105.0, rel=1e-2), "Missing price should be interpolated to approximately 105"

def test_preprocess_data_all_missing_prices():
    """
    Test preprocess_data with all price values missing and expect a ValueError.
    """
    # Create a sample raw DataFrame with all prices missing
    raw_df = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
        'price': [np.nan, np.nan]
    })

    with pytest.raises(ValueError, match="After interpolation and dropping NaNs, no rows left."):
        preprocess_data(raw_df)

def test_preprocess_data_missing_columns():
    """
    Test preprocess_data with missing required columns and expect a KeyError.
    """
    # Create a sample raw DataFrame missing 'price'
    raw_df = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
        'volume': [1000, 1100]
    })

    with pytest.raises(KeyError, match="Missing required columns: {'price'}"):
        preprocess_data(raw_df)

def test_preprocess_data_log_return_calculation():
    """
    Test preprocess_data to ensure log_return is calculated correctly.
    """
    # Create a sample raw DataFrame
    raw_df = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']),
        'price': [100, 200, 400]
    })

    # Call the preprocessing function
    processed_df = preprocess_data(raw_df)

    # Expected log returns
    expected_log_return_1 = np.log(200 / 100)
    expected_log_return_2 = np.log(400 / 200)

    # Assertions
    np.testing.assert_almost_equal(processed_df['log_return'].iloc[0], expected_log_return_1, decimal=5)
    np.testing.assert_almost_equal(processed_df['log_return'].iloc[1], expected_log_return_2, decimal=5)

# ------------------------------------------------------------------------------
# Tests for train_test_split
# ------------------------------------------------------------------------------

def test_train_test_split_normal():
    """
    Test train_test_split with normal data and default train_ratio.
    """
    # Create a sample DataFrame with 10 rows
    prices = np.arange(100, 110)
    prices_series = pd.Series(prices)
    log_returns = np.log(prices_series / prices_series.shift(1)).dropna()

    df = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
        'price': prices,
        'log_return': log_returns
    })

    # Call the split function
    train_df, test_df = train_test_split(df)

    # Assertions
    assert len(train_df) == 8, "Train set should have 8 rows (80%)"
    assert len(test_df) == 2, "Test set should have 2 rows (20%)"
    assert train_df.iloc[-1]['price'] == 107, "Last train price should be 107"
    assert test_df.iloc[0]['price'] == 108, "First test price should be 108"

def test_train_test_split_custom_ratio():
    """
    Test train_test_split with a custom train_ratio.
    """
    # Create a sample DataFrame with 5 rows
    prices = np.array([100, 105, 110, 115, 120])
    prices_series = pd.Series(prices)
    log_returns = np.log(prices_series / prices_series.shift(1)).dropna()

    df = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
        'price': prices,
        'log_return': log_returns
    })

    # Call the split function with train_ratio=0.6
    train_df, test_df = train_test_split(df, train_ratio=0.6)

    # Assertions
    assert len(train_df) == 3, "Train set should have 3 rows (60%)"
    assert len(test_df) == 2, "Test set should have 2 rows (40%)"
    assert train_df.iloc[-1]['price'] == 110, "Last train price should be 110"
    assert test_df.iloc[0]['price'] == 115, "First test price should be 115"

def test_train_test_split_invalid_ratio():
    """
    Test train_test_split with invalid train_ratio values and expect a ValueError.
    """
    # Create a sample DataFrame
    prices = np.arange(100, 105)
    prices_series = pd.Series(prices)
    log_returns = np.log(prices_series / prices_series.shift(1)).dropna()

    df = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
        'price': prices,
        'log_return': log_returns
    })

    # Test with train_ratio=0
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1."):
        train_test_split(df, train_ratio=0)

    # Test with train_ratio=1
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1."):
        train_test_split(df, train_ratio=1)

    # Test with negative train_ratio
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1."):
        train_test_split(df, train_ratio=-0.1)

def test_train_test_split_small_dataframe():
    """
    Test train_test_split with a DataFrame that's too small to split.
    Expect a ValueError.
    """
    # Create a sample DataFrame with only one row
    df = pd.DataFrame({
        'date': [pd.to_datetime('2022-01-01')],
        'price': [100],
        'log_return': [0.0]
    })

    with pytest.raises(ValueError, match="DataFrame too small"):
        train_test_split(df, train_ratio=0.8)

# ------------------------------------------------------------------------------
# Tests for compute_descriptive_stats
# ------------------------------------------------------------------------------

def test_compute_descriptive_stats_full():
    """
    Test compute_descriptive_stats with both 'price' and 'log_return' columns.
    """
    # Create a sample DataFrame
    df = pd.DataFrame({
        'price': [100, 200, 300, 400, 500],
        'log_return': [0.0, 0.693147, 0.405465, 0.287682, 0.223144]
    })

    # Call the stats function
    stats = compute_descriptive_stats(df)

    # Assertions
    assert 'price_mean' in stats, "'price_mean' should be in stats"
    assert 'logret_mean' in stats, "'logret_mean' should be in stats"
    assert stats['price_mean'] == 300.0, "Mean price should be 300.0"
    assert stats['logret_mean'] == pytest.approx(0.321988, rel=1e-5), "Mean log_return should be approx 0.321988"

def test_compute_descriptive_stats_price_only():
    """
    Test compute_descriptive_stats with only the 'price' column.
    """
    # Create a sample DataFrame
    df = pd.DataFrame({
        'price': [50, 60, 70, 80, 90]
    })

    # Call the stats function
    stats = compute_descriptive_stats(df)

    # Assertions
    assert 'price_mean' in stats, "'price_mean' should be in stats"
    assert 'logret_mean' not in stats, "'logret_mean' should not be in stats"

def test_compute_descriptive_stats_log_return_only():
    """
    Test compute_descriptive_stats with only the 'log_return' column.
    """
    # Create a sample DataFrame
    df = pd.DataFrame({
        'log_return': [0.1, 0.2, 0.3, 0.4]
    })

    # Call the stats function
    stats = compute_descriptive_stats(df)

    # Assertions
    assert 'price_mean' not in stats, "'price_mean' should not be in stats"
    assert 'logret_mean' in stats, "'logret_mean' should be in stats"

def test_compute_descriptive_stats_missing_columns():
    """
    Test compute_descriptive_stats with no relevant columns and expect an empty dict.
    """
    # Create a sample DataFrame with irrelevant columns
    df = pd.DataFrame({
        'volume': [1000, 2000, 3000]
    })

    # Call the stats function
    stats = compute_descriptive_stats(df)

    # Assertions
    assert stats == {}, "Stats dict should be empty when no relevant columns are present"

# ------------------------------------------------------------------------------
# Tests for fetch_data_for_coins
# ------------------------------------------------------------------------------

@patch('data_loader.fetch_data_yahoo')
@patch('data_loader.preprocess_data')
def test_fetch_data_for_coins_valid(mock_preprocess, mock_fetch_data_yahoo):
    """
    Test fetch_data_for_coins with valid coin_ids and ensure data is fetched and processed correctly.
    """
    # Setup mock return values
    mock_fetch_data_yahoo.return_value = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
        'price': [100, 105]
    })
    mock_preprocess.return_value = pd.DataFrame({
        'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
        'price': [100, 105],
        'log_return': [0.0, 0.048790164]
    })

    # Define the coins to fetch
    coins = ['bitcoin', 'ethereum']

    # Call the function
    result = fetch_data_for_coins(coins, start='2022-01-01', end='2022-01-02')

    # Assertions
    assert isinstance(result, dict), "Result should be a dictionary"
    assert set(result.keys()) == set(coins), "Result keys should match the input coin_ids"
    for coin in coins:
        df = result[coin]
        assert 'date' in df.columns and 'price' in df.columns and 'log_return' in df.columns, \
            f"DataFrame for {coin} should have 'date', 'price', and 'log_return' columns"
        assert len(df) == 2, f"DataFrame for {coin} should have 2 rows"

def test_fetch_data_for_coins_invalid_coin():
    """
    Test fetch_data_for_coins with an invalid coin_id and expect a ValueError.
    """
    # Define the coins to fetch, including an invalid one
    coins = ['bitcoin', 'invalidcoin']

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Unknown coin_id 'invalidcoin'"):
        fetch_data_for_coins(coins, start='2022-01-01', end='2022-01-02')

@patch('data_loader.fetch_data_yahoo')
@patch('data_loader.preprocess_data')
def test_fetch_data_for_coins_empty_result(mock_preprocess, mock_fetch_data_yahoo):
    """
    Test fetch_data_for_coins when fetch_data_yahoo returns empty DataFrames.
    """
    # Setup mock return values
    mock_fetch_data_yahoo.return_value = pd.DataFrame()
    mock_preprocess.side_effect = ValueError("No data returned for BTC-USD")

    # Define the coins to fetch
    coins = ['bitcoin']

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="No data returned for BTC-USD"):
        fetch_data_for_coins(coins, start='2022-01-01', end='2022-01-02')

# ------------------------------------------------------------------------------
# End of Test Suite
# ------------------------------------------------------------------------------