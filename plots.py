"""
plots.py

Generate Plotly charts for the Dash application.

Author: Your Name
"""

import plotly.graph_objs as go
import pandas as pd

def price_plot(df: pd.DataFrame, forecast_df: pd.DataFrame = None):
    """
    Create a line plot of actual vs forecasted prices.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with 'date' and 'price'.
    forecast_df : pd.DataFrame
        Forecasted data with 'date' and 'forecast_price'.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines', name='Actual Price'))
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast_price'], 
                                 mode='lines', name='Forecasted Price'))
    fig.update_layout(title='Actual vs Forecasted Price', xaxis_title='Date', yaxis_title='Price')
    return fig