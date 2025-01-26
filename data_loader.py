"""
data_loader.py

Handles data retrieval from Yahoo Finance, basic preprocessing
(cleaning, interpolation, log returns), and optional splitting.

Author: Markus Öffel
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict, List

# ------------------------------------------------------------------------------
# 1) CONSTANTS AND CONFIGURATIONS
# ------------------------------------------------------------------------------
# CRYPTO_SYMBOLS maps common cryptocurrency names to their Yahoo Finance ticker symbols
CRYPTO_SYMBOLS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "solana":  "SOL-USD",
    # "litecoin": "LTC-USD",  # Uncomment if needed
    # "ripple":   "XRP-USD",  # Uncomment if needed
}

# ------------------------------------------------------------------------------
# 2) DATA FETCHING FUNCTIONS
# ------------------------------------------------------------------------------
def fetch_data_yahoo(coin_id: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical daily price data for a specified cryptocurrency from Yahoo Finance.
    Flattens multi-level columns if needed (e.g., 'Close_BTC-USD' -> 'price').

    Parameters
    ----------
    coin_id : str
        The identifier for the cryptocurrency (e.g., "bitcoin", "ethereum").
        Must be a key in the CRYPTO_SYMBOLS dictionary.
    start : str, optional
        The start date for data retrieval in 'YYYY-MM-DD' format. If None, defaults to the earliest available date.
    end : str, optional
        The end date for data retrieval in 'YYYY-MM-DD' format. If None, defaults to today's date.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing two columns:
            - 'date': The date of the price record.
            - 'price': The closing price of the cryptocurrency on that date.

    Raises
    ------
    ValueError
        If the provided coin_id is not recognized or if data retrieval fails.
    KeyError
        If the expected 'Close' column is not found in the downloaded data.

    Examples
    --------
    >>> df = fetch_data_yahoo("bitcoin", start="2022-01-01", end="2022-01-10")
    >>> df.shape[0]  # Number of rows fetched
    10
    >>> df.columns.tolist()
    ['date', 'price']
    >>> df.iloc[0]['price']  # Example price value
    46200.0  # (Note: Actual value may vary)
    """
    # Validate the provided coin_id exists in the CRYPTO_SYMBOLS dictionary
    if coin_id not in CRYPTO_SYMBOLS:
        raise ValueError(f"Unknown coin_id '{coin_id}'. Please add it to CRYPTO_SYMBOLS.")

    def clean_date(date_str: str) -> str:
        """
        Cleans the input date string by removing any time component.

        Parameters
        ----------
        date_str : str
            The date string to clean.

        Returns
        -------
        str
            The cleaned date string in 'YYYY-MM-DD' format.

        Examples
        --------
        >>> clean_date("2022-01-01T00:00:00")
        '2022-01-01'
        >>> clean_date("2022-01-01")
        '2022-01-01'
        >>> clean_date(None) is None
        True
        """
        if date_str:
            if 'T' in date_str:
                return date_str.split('T')[0]
            return date_str
        return None

    # Clean the start and end dates
    start_clean = clean_date(start)
    end_clean   = clean_date(end)
    ticker_symbol = CRYPTO_SYMBOLS[coin_id]

    try:
        # Download historical data from Yahoo Finance
        data = yf.download(
            ticker_symbol,
            start=start_clean,
            end=end_clean,
            progress=False,
            group_by="column"
        )
    except Exception as e:
        # Handle any exceptions that occur during data download
        raise ValueError(f"Failed to download data from yfinance: {str(e)}")

    # Check if any data was returned
    if data.empty:
        raise ValueError(
            f"No data returned for {ticker_symbol} between "
            f"{start_clean or 'start'} and {end_clean or 'now'}"
        )

    # Flatten columns if they are multi-indexed (e.g., 'Close_BTC-USD')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(level) for level in col if level])
            for col in data.columns.to_flat_index()
        ]

    # Identify the 'Close' column
    close_cols = [c for c in data.columns if c.startswith("Close")]
    if not close_cols:
        raise KeyError(f"No 'Close' column found in downloaded data. Columns: {data.columns.tolist()}")
    close_col = close_cols[0]

    # Reset index to move 'Date' from index to a column
    data = data.reset_index()
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    data.rename(columns={close_col: "price"}, inplace=True)

    # Ensure required columns are present
    if not {"date", "price"}.issubset(data.columns):
        missing = {"date", "price"} - set(data.columns)
        raise KeyError(f"Missing required columns: {missing}")

    # Return the cleaned DataFrame sorted by date
    return data[["date", "price"]].sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------------------------
# 3) DATA PREPROCESSING FUNCTIONS
# ------------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Cleaning: 
      1) Sort by date
      2) Interpolate missing prices
      3) Drop any remaining NaN in 'price'
      4) Compute log_return = ln(price[t]/price[t-1])
      5) Drop the first row of log_return which is NaN
      6) Reset index

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame containing 'date' and 'price' columns.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with 'date', 'price', and 'log_return' columns.

    Raises
    ------
    KeyError
        If the required columns are missing from the input DataFrame.
    ValueError
        If no data remains after interpolation and dropping NaNs.

    Examples
    --------
    >>> raw_df = pd.DataFrame({
    ...     'date': ['2022-01-01', '2022-01-02', '2022-01-04'],
    ...     'price': [100, np.nan, 110]
    ... })
    >>> processed_df = preprocess_data(raw_df)
    >>> processed_df.shape[0]
    2
    >>> 'log_return' in processed_df.columns
    True
    >>> round(processed_df.iloc[0]['log_return'], 5)
    0.09531  # Approximately ln(110/100)
    """
    required_cols = {"date", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Sort the DataFrame by date in ascending order
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # 1) Interpolate missing price values linearly
    #    (You could use method='time' if 'date' is a DateTimeIndex,
    #     or other interpolation methods as needed.)
    df["price"] = df["price"].interpolate(method="linear", limit_direction="both")

    # 2) Drop any rows still having NaN after interpolation (leading/trailing NAs)
    df.dropna(subset=["price"], inplace=True)
    if df.empty:
        raise ValueError("After interpolation and dropping NaNs, no rows left.")

    # 3) Compute log returns
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    # 4) Drop the first row with NaN in log_return
    df.dropna(subset=["log_return"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------------------------------------------------------
# 4) TRAIN-TEST SPLIT FUNCTIONS
# ------------------------------------------------------------------------------
def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train/test sets by time index according to train_ratio.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame to split.
    train_ratio : float, optional
        The proportion of the data to include in the training set (default is 0.8).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training DataFrame and the testing DataFrame.

    Raises
    ------
    ValueError
        If the train_ratio is not between 0 and 1 or if the DataFrame is too small to split.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
    ...     'price': np.arange(100, 110),
    ...     'log_return': np.log(np.arange(100, 110) / np.arange(100, 110).shift(1))
    ... }).dropna()
    >>> train_df, test_df = train_test_split(df, train_ratio=0.7)
    >>> len(train_df)
    7
    >>> len(test_df)
    3
    >>> train_df.iloc[-1]['price']
    106
    >>> test_df.iloc[0]['price']
    107
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if len(df) < 2:
        raise ValueError(f"DataFrame too small ({len(df)}) to split.")

    # Determine the index at which to split the DataFrame
    split_index = int(len(df) * train_ratio)

    # Split the DataFrame into training and testing sets
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df  = df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df

# ------------------------------------------------------------------------------
# 5) DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------
def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes descriptive statistics for 'price' and 'log_return':
      - mean, std, min, max, skewness, kurtosis

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with columns 'price' and optionally 'log_return'.

    Returns
    -------
    Dict[str, float]
        A dictionary like {
          'price_mean', 'price_std', 'price_min', 'price_max', 
          'price_skew', 'price_kurtosis', 'logret_mean', ...
        }

    Examples
    --------
    >>> import pandas as pd
    >>> df_stats = pd.DataFrame({
    ...     'price': [100, 105, 110, 120],
    ...     'log_return': [0.0, 0.048790164, 0.046520015, 0.087011373]
    ... })
    >>> stats = compute_descriptive_stats(df_stats)
    >>> round(stats['price_mean'], 2)
    108.75
    >>> round(stats['logret_std'], 6)
    0.024046
    >>> 'logret_skew' in stats
    True
    """
    stats_dict = {}

    if 'price' in df.columns:
        ser_p = df['price'].dropna()
        stats_dict['price_mean'] = ser_p.mean()
        stats_dict['price_std'] = ser_p.std()
        stats_dict['price_min'] = ser_p.min()
        stats_dict['price_max'] = ser_p.max()
        stats_dict['price_skew'] = ser_p.skew()
        stats_dict['price_kurtosis'] = ser_p.kurtosis()

    if 'log_return' in df.columns:
        ser_lr = df['log_return'].dropna()
        stats_dict['logret_mean'] = ser_lr.mean()
        stats_dict['logret_std'] = ser_lr.std()
        stats_dict['logret_min'] = ser_lr.min()
        stats_dict['logret_max'] = ser_lr.max()
        stats_dict['logret_skew'] = ser_lr.skew()
        stats_dict['logret_kurtosis'] = ser_lr.kurtosis()

    # Convert to float to avoid numpy dtypes in the dictionary
    stats_dict = {k: float(v) for k, v in stats_dict.items()}
    return stats_dict

# ------------------------------------------------------------------------------
# 6) FETCH MULTIPLE COINS
# ------------------------------------------------------------------------------
def fetch_data_for_coins(coin_ids: List[str],
                         start: str = "2019-01-01",
                         end: str = None) -> Dict[str, pd.DataFrame]:
    """
    Loads and preprocesses data for multiple coins from Yahoo Finance.

    For each coin in coin_ids, we call fetch_data_yahoo(...) and then preprocess_data(...).
    We store the resulting DataFrame in a dictionary under the coin_id key.

    Parameters
    ----------
    coin_ids : List[str]
        e.g. ["bitcoin", "ethereum"]. Must be in CRYPTO_SYMBOLS.
    start : str, optional
        Start date, default '2019-01-01'.
    end : str, optional
        End date, default None => today's date.

    Returns
    -------
    Dict[str, pd.DataFrame]
        e.g. {
          'bitcoin':  <DataFrame with date, price, log_return>,
          'ethereum': <...>,
          ...
        }

    Raises
    ------
    ValueError
        If one of the coin_ids is not recognized or if no data returned (caught in fetch_data_yahoo).
    KeyError
        If required columns are missing (caught in preprocess_data).

    Examples
    --------
    (We'll skip actual doctests because yfinance can vary data. We'll just do a demonstration snippet.)

    # Example (no test assertion):
    # >>> coins = ['bitcoin', 'ethereum']
    # >>> big_dict = fetch_data_for_coins(coins, start='2023-01-01', end='2023-03-01')
    # >>> list(big_dict.keys())  # doctest: +SKIP
    # ['bitcoin', 'ethereum']
    # >>> big_dict['bitcoin'].columns.tolist()  # doctest: +SKIP
    # ['date', 'price', 'log_return']
    """
    all_data = {}
    for cid in coin_ids:
        # Fetch raw data for the current coin
        df_raw = fetch_data_yahoo(cid, start=start, end=end)
        # Preprocess the fetched data
        df_proc = preprocess_data(df_raw)
        # Store the processed DataFrame in the dictionary
        all_data[cid] = df_proc
    return all_data

# ------------------------------------------------------------------------------
# 7) MAIN TEST/DEMO (when run as standalone script)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest

    print("Running doctests in 'data_loader.py' ...")
    # Execute all doctests in the module
    results = doctest.testmod()
    if results.failed:
        print(f"\n*** {results.failed} doctest(s) failed ***")
    else:
        print("\nAll doctests passed successfully.")

    # Quick manual test to demonstrate functionality
    print("\n=== Manual Test ===")
    try:
        # Example: Load data for Bitcoin from 2022-01-01 to the current date
        df_btc = fetch_data_yahoo("bitcoin", start="2022-01-01", end=None)
        print(f"BTC rows fetched: {len(df_btc)}")
        print("First 5 rows:\n", df_btc.head())
        print("Last 5 rows:\n", df_btc.tail())

        # Preprocess the fetched Bitcoin data
        df_btc = preprocess_data(df_btc)
        print(f"After preprocess, rows: {len(df_btc)}")
        if not df_btc.empty:
            # Compute descriptive statistics for the preprocessed data
            stats_btc = compute_descriptive_stats(df_btc)
            print("Descriptive stats BTC:", stats_btc)
        else:
            print("No data left after preprocessing; can't compute stats.")

        # Split the data into training and testing sets
        train_btc, test_btc = train_test_split(df_btc, train_ratio=0.8)
        print(f"Train size: {len(train_btc)}, Test size: {len(test_btc)}")

    except Exception as e:
        # Catch and display any errors that occur during the manual test
        print("ERROR in manual test:", e)