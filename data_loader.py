"""
data_loader.py

Handles data retrieval (e.g., from CoinGecko or Yahoo Finance) 
and basic preprocessing (cleaning, log returns, splitting).

Author: Your Name
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Tuple

def fetch_data_coingecko(coin_id: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch historical daily price data for a specified cryptocurrency from CoinGecko.

    Parameters
    ----------
    coin_id : str
        The CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'dogecoin', etc.)
    days : int
        Number of past days of data to fetch (e.g., 365 for 1 year).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'price', 'date'].

    Notes
    -----
    - The CoinGecko API returns timestamp in milliseconds and a daily price.
    - If you exceed API limits, consider adding a retry or caching mechanism.

    Examples
    --------
    >>> df = fetch_data_coingecko('bitcoin', 30)
    >>> isinstance(df, pd.DataFrame)
    True
    """
    # Construct the API endpoint
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    # Example: https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365
    params = {
        "vs_currency": "usd",
        "days": days
    }

    # Send GET request to CoinGecko API
    response = requests.get(url, params=params)
    data = response.json()

    # The "prices" key in the returned JSON is a list of [timestamp_ms, price]
    prices = data["prices"]  # e.g., [[1638316800000, 57000], [1638403200000, 58000], ...]

    # Convert to a DataFrame
    df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])

    # Convert timestamp from milliseconds to a datetime
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit='ms')

    # Drop the original timestamp_ms column or rename it
    df.rename(columns={"timestamp_ms": "timestamp"}, inplace=True)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw data: sorts by date, drops missing values,
    and calculates log returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ['date', 'price'] columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with additional column ['log_return'].

    Examples
    --------
    >>> sample_df = pd.DataFrame({
    ...     'date': pd.date_range(start='2021-01-01', periods=3),
    ...     'price': [30000, 31000, 32000]
    ... })
    >>> out_df = preprocess_data(sample_df)
    >>> 'log_return' in out_df.columns
    True
    """
    # Sort values by date to ensure time-series order
    df.sort_values("date", inplace=True)

    # Remove any rows with NaN values
    df.dropna(subset=["price"], inplace=True)

    # Compute the log returns: log( price[t] / price[t-1] )
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    # Remove the first row which will have NaN from shift
    df.dropna(inplace=True)

    return df


def train_test_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train and test sets based on the specified ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time series data (already preprocessed).
    train_ratio : float
        The fraction of the data to use for training (default is 0.80).

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        - train_df: Training portion of the data
        - test_df: Testing portion of the data

    Examples
    --------
    >>> sample_df = pd.DataFrame({
    ...     'date': pd.date_range(start='2021-01-01', periods=5),
    ...     'price': [30000, 31000, 32000, 33000, 34000],
    ...     'log_return': [None, 0.03, 0.032, 0.031, 0.03]
    ... }).dropna()
    >>> train_df, test_df = train_test_split(sample_df, train_ratio=0.6)
    >>> len(train_df) < len(sample_df)
    True
    >>> len(test_df) > 0
    True
    """
    # Calculate the index at which we split
    split_index = int(len(df) * train_ratio)
    
    # Split the data
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df