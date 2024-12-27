"""
data_loader.py

Handles data retrieval from Yahoo Finance, basic preprocessing
(cleaning, log returns, splitting), and computes descriptive statistics.

This module is designed so that we can fetch historical cryptocurrency prices
beyond the 365-day limit that CoinGecko imposes for free users.

Now includes logic to flatten multi-level columns from yfinance (e.g., if it 
returns 'Close_BTC-USD' instead of just 'Close').

Author: Your Name
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict, List

# ------------------------------------------------------------------------------
# 1) MAPPING: CRYPTO_SYMBOLS
# ------------------------------------------------------------------------------
# You can add more cryptocurrencies as needed, linking your internal "coin_id"
# to the appropriate Yahoo Finance ticker symbol (e.g., "BTC-USD", "ETH-USD").
CRYPTO_SYMBOLS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "solana": "SOL-USD",
    # Example extension:
    # "litecoin": "LTC-USD",
    # "ripple": "XRP-USD",
}

# ------------------------------------------------------------------------------
# 2) FETCHING DATA FROM YAHOO FINANCE (with column-flattening)
# ------------------------------------------------------------------------------
def fetch_data_yahoo(coin_id: str, start: str = "2019-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch historical daily price data for a specified cryptocurrency from Yahoo Finance.

    This function:
      - Uses group_by="column" so we don't get ticker-labeled columns.
      - If MultiIndex columns remain, we flatten them.
      - Renames "Close" to "price" and ensures we have columns: ['date', 'price'].

    Parameters
    ----------
    coin_id : str
        A key identifying which crypto to fetch, e.g. 'bitcoin', 'ethereum'.
        Must exist in the CRYPTO_SYMBOLS mapping above.
    start : str, optional
        Start date in 'YYYY-MM-DD' format, defaulting to '2019-01-01'.
    end : str, optional
        End date in 'YYYY-MM-DD', default None => today's date.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['date', 'price'] (plus any others if you want),
        sorted ascending by 'date'.

    Raises
    ------
    ValueError
        If the coin_id is unknown or if no data is returned from Yahoo Finance.
    KeyError
        If the 'Close' column cannot be found after flattening.

    Examples
    --------
    >>> df_test = fetch_data_yahoo('bitcoin', start='2022-01-01', end='2022-01-05')  # doctest: +SKIP
    >>> 'date' in df_test.columns                                                    # doctest: +SKIP
    True
    >>> 'price' in df_test.columns                                                   # doctest: +SKIP
    True
    """
    # Ensure that the coin_id is known
    if coin_id not in CRYPTO_SYMBOLS:
        raise ValueError(f"Unknown coin_id '{coin_id}'. Please add it to CRYPTO_SYMBOLS.")

    ticker_symbol = CRYPTO_SYMBOLS[coin_id]

    # Download data from Yahoo with group_by="column"
    # so that we don't have ticker-labeled columns.
    data = yf.download(ticker_symbol, start=start, end=end,
                       progress=False, group_by="column")

    if data.empty:
        raise ValueError(
            f"No data returned from Yahoo Finance for '{ticker_symbol}' "
            f"in the range {start} to {end}."
        )

    # If columns are multi-indexed, flatten them
    # e.g. ('Close', 'BTC-USD') => 'Close_BTC-USD'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(level) for level in col if level])
            for col in data.columns.to_flat_index()
        ]

    # We look for a column that starts with "Close". In many cases, after flattening,
    # you might get "Close" or "Close_BTC-USD". Let's find that column:
    close_cols = [c for c in data.columns if c.startswith("Close")]
    if not close_cols:
        raise KeyError(
            f"No 'Close' column found after flattening. Columns are: {data.columns.tolist()}"
        )

    # We'll take the first such column as the "price" column
    close_col = close_cols[0]

    # We assume there's a 'Date' index or column
    data = data.reset_index()

    # Rename that index/column to 'date' if it's named 'Date'
    # (In some environments it might already be a column named 'Date'.)
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    elif "index" in data.columns:
        data.rename(columns={"index": "date"}, inplace=True)

    # Now rename the close_col to "price"
    data.rename(columns={close_col: "price"}, inplace=True)

    # We want a final DataFrame with at least ['date', 'price']
    if "date" not in data.columns or "price" not in data.columns:
        raise KeyError(
            f"Missing required columns after rename. 'date' or 'price' not found. "
            f"Columns are: {data.columns.tolist()}"
        )

    # Filter to just ['date', 'price'] if you only want those
    # or keep other columns if you want them. For now let's do just 2:
    df = data[["date", "price"]].copy()

    # Sort by date
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------------------------------------------------------
# 3) PREPROCESSING: Sorting, Dropping NA, Log-Returns
# ------------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw price data by:
      1. Dropping any rows with NaN in 'price'.
      2. Calculating the log returns: log(price[t] / price[t-1]).
      3. Dropping the row where log_return is NaN.
      4. Resetting the index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ['date', 'price'] columns.

    Returns
    -------
    pd.DataFrame
        The same data but with an additional column 'log_return'.

    Raises
    ------
    KeyError
        If 'date' or 'price' columns are missing.
    ValueError
        If data is entirely NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> sample_df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=5),
    ...     'price': [20000, 20100, None, 20500, 21000]
    ... })
    >>> out_df = preprocess_data(sample_df)
    >>> 'log_return' in out_df.columns
    True
    """
    required_cols = {'date', 'price'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Drop rows where price is NaN
    df_cleaned = df.dropna(subset=["price"]).copy()
    if df_cleaned.empty:
        raise ValueError("After dropping NaN prices, no rows left.")

    # Compute log returns
    df_cleaned["log_return"] = np.log(df_cleaned["price"] / df_cleaned["price"].shift(1))

    # Drop the row with NaN in log_return (the first one)
    df_cleaned.dropna(subset=["log_return"], inplace=True)

    # Reset index
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

# ------------------------------------------------------------------------------
# 4) TRAIN-TEST SPLIT
# ------------------------------------------------------------------------------
def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-series DataFrame into train and test sets, based on a ratio.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed data with 'price' and 'log_return'.
    train_ratio : float, optional
        The fraction of data for training. Default=0.8.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        train_df, test_df

    Raises
    ------
    ValueError
        If train_ratio is <= 0 or >= 1, or if df is too small.

    Examples
    --------
    >>> import pandas as pd
    >>> df_example = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=6),
    ...     'price': [1,2,3,4,5,6],
    ...     'log_return': [0.0]*6
    ... })
    >>> train_df, test_df = train_test_split(df_example, train_ratio=0.5)
    >>> len(train_df)
    3
    >>> len(test_df)
    3
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if len(df) < 2:
        raise ValueError(f"DataFrame too small ({len(df)}) to split.")

    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
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
    start : str
        Start date, default '2019-01-01'.
    end : str
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
    (We'll skip actual doc test because yfinance can vary data. We'll just do a demonstration snippet.)

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
        df_raw = fetch_data_yahoo(cid, start=start, end=end)
        df_proc = preprocess_data(df_raw)
        all_data[cid] = df_proc
    return all_data

# ------------------------------------------------------------------------------
# 7) MAIN TEST/DEMO (when run as standalone script)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest

    print("Running doctests in 'data_loader.py' ...")
    results = doctest.testmod()
    if results.failed:
        print(f"\n*** {results.failed} doctest(s) failed ***")
    else:
        print("\nAll doctests passed successfully.")

    # Quick manual test
    print("\n=== Manual Test ===")
    try:
        # Example: Load data for BTC from 2022-01-01 to the current date
        df_btc = fetch_data_yahoo("bitcoin", start="2022-01-01", end=None)
        print(f"BTC rows fetched: {len(df_btc)}")
        print("First 5 rows:\n", df_btc.head())
        print("Last 5 rows:\n", df_btc.tail())

        # Preprocess
        df_btc = preprocess_data(df_btc)
        print(f"After preprocess, rows: {len(df_btc)}")
        if not df_btc.empty:
            stats_btc = compute_descriptive_stats(df_btc)
            print("Descriptive stats BTC:", stats_btc)
        else:
            print("No data left after preprocessing; can't compute stats.")

        # Train/Test split
        train_btc, test_btc = train_test_split(df_btc, train_ratio=0.8)
        print(f"Train size: {len(train_btc)}, Test size: {len(test_btc)}")

    except Exception as e:
        print("ERROR in manual test:", e)