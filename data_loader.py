"""
data_loader.py

Handles data retrieval from Yahoo Finance, basic preprocessing
(cleaning, interpolation, log returns), and optional splitting.

Author: Your Name
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict, List

# CRYPTO_SYMBOLS for Yahoo Finance
CRYPTO_SYMBOLS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "solana":  "SOL-USD",
    # "litecoin": "LTC-USD",
    # "ripple":   "XRP-USD",
}

def fetch_data_yahoo(coin_id: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical daily price data for a specified cryptocurrency from Yahoo Finance.
    Flattens multi-level columns if needed (Close_BTC-USD -> price).
    """
    # Validate
    if coin_id not in CRYPTO_SYMBOLS:
        raise ValueError(f"Unknown coin_id '{coin_id}'. Please add it to CRYPTO_SYMBOLS.")

    def clean_date(date_str: str) -> str:
        if date_str:
            if 'T' in date_str:
                return date_str.split('T')[0]
            return date_str
        return None

    start_clean = clean_date(start)
    end_clean   = clean_date(end)
    ticker_symbol = CRYPTO_SYMBOLS[coin_id]

    try:
        data = yf.download(
            ticker_symbol,
            start=start_clean,
            end=end_clean,
            progress=False,
            group_by="column"
        )
    except Exception as e:
        raise ValueError(f"Failed to download data from yfinance: {str(e)}")

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker_symbol} between "
            f"{start_clean or 'start'} and {end_clean or 'now'}"
        )

    # Flatten columns (in case of multi-index)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(level) for level in col if level])
            for col in data.columns.to_flat_index()
        ]

    # Search for a 'Close' column
    close_cols = [c for c in data.columns if c.startswith("Close")]
    if not close_cols:
        raise KeyError(f"No 'Close' column found in downloaded data. Columns: {data.columns.tolist()}")
    close_col = close_cols[0]

    data = data.reset_index()
    if "Date" in data.columns:
        data.rename(columns={"Date": "date"}, inplace=True)
    data.rename(columns={close_col: "price"}, inplace=True)

    if not {"date", "price"}.issubset(data.columns):
        missing = {"date", "price"} - set(data.columns)
        raise KeyError(f"Missing required columns: {missing}")

    return data[["date", "price"]].sort_values("date").reset_index(drop=True)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Cleaning: 
      1) Sort by date
      2) Interpolate missing prices
      3) Drop any remaining NaN in 'price'
      4) Compute log_return = ln(price[t]/price[t-1])
      5) Drop the first row of log_return which is NaN
      6) Reset index
    """
    required_cols = {"date", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Sort by date (if not already sorted)
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

def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train/test sets by time index according to train_ratio.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if len(df) < 2:
        raise ValueError(f"DataFrame too small ({len(df)}) to split.")

    split_index = int(len(df) * train_ratio)
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