"""
plots.py

Generate various Plotly charts for the Dash application,
plus HTML tables for descriptive stats and forecasted prices.

Author: Your Name
"""

import matplotlib

# If you want to avoid GUI issues on macOS, uncomment the following:
matplotlib.use("Agg")

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash import html

# (Optional) If using ACF/PACF or Q-Q plots, you'll need these:
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def price_plot(df: pd.DataFrame, forecast_df: pd.DataFrame = None):
    """
    Create a line plot of actual vs forecasted prices.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with ['date', 'price'].
    forecast_df : pd.DataFrame, optional
        Forecasted data with ['date', 'forecast_price'].

    Returns
    -------
    plotly.graph_objs.Figure
    """
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Actual Price'
    ))

    # Forecast
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast_price'],
            mode='lines',
            name='Forecasted Price'
        ))

    fig.update_layout(
        title='Actual vs Forecasted Price',
        xaxis_title='Date',
        yaxis_title='Price (USD)'
    )
    return fig


def histogram_plot(df: pd.DataFrame):
    """
    Creates a histogram for the 'log_return' column in df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'log_return'.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['log_return'], nbinsx=50, name='Log Returns'))
    fig.update_layout(
        title='Histogram of Log Returns',
        xaxis_title='Log Return',
        yaxis_title='Count'
    )
    return fig


def qq_plot(df: pd.DataFrame):
    """
    Creates a Q-Q plot for the 'log_return' column using statsmodels
    and converts it into a Plotly Figure.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    log_returns = df['log_return'].dropna()
    qq = sm.ProbPlot(log_returns, fit=True)
    theoretical_quantiles = qq.theoretical_quantiles
    sample_quantiles = qq.sample_quantiles

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Data'
    ))
    # Add 45-degree line
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=theoretical_quantiles,
        mode='lines',
        name='45-line'
    ))
    fig.update_layout(
        title='Q-Q Plot (Log Returns)',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles'
    )
    return fig


import base64

def acf_plot(df: pd.DataFrame, lags=40):
    """
    ACF plot for 'log_return', converted to a Base64-encoded image for Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the 'log_return' column.
    lags : int
        Number of lags to show in the ACF plot.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    import io
    from statsmodels.graphics.tsaplots import plot_acf

    # Create ACF plot with Matplotlib
    fig_mpl = plot_acf(df['log_return'].dropna(), lags=lags)
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)  # Move to the beginning of the buffer
    plt.close(fig_mpl)

    # Encode the image as Base64
    base64_image = base64.b64encode(buf.read()).decode('utf-8')

    # Create Plotly figure
    fig_plotly = go.Figure()
    fig_plotly.add_layout_image(
        dict(
            source=f"data:image/png;base64,{base64_image}",
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            sizex=1,
            sizey=1,
            xanchor="left",
            yanchor="top",
            layer="below",
            sizing="contain"
        )
    )
    fig_plotly.update_layout(
        width=700,
        height=400,
        title="ACF Plot"
    )
    return fig_plotly


def pacf_plot(df: pd.DataFrame, lags=40):
    """
    PACF plot for 'log_return', converted to a Base64-encoded image for Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the 'log_return' column.
    lags : int
        Number of lags to show in the PACF plot.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    import io
    from statsmodels.graphics.tsaplots import plot_pacf

    # Create PACF plot with Matplotlib
    fig_mpl = plot_pacf(df['log_return'].dropna(), lags=lags, method='yw')
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)  # Move to the beginning of the buffer
    plt.close(fig_mpl)

    # Encode the image as Base64
    base64_image = base64.b64encode(buf.read()).decode('utf-8')

    # Create Plotly figure
    fig_plotly = go.Figure()
    fig_plotly.add_layout_image(
        dict(
            source=f"data:image/png;base64,{base64_image}",
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            sizex=1,
            sizey=1,
            xanchor="left",
            yanchor="top",
            layer="below",
            sizing="contain"
        )
    )
    fig_plotly.update_layout(
        width=700,
        height=400,
        title="PACF Plot"
    )
    return fig_plotly


def create_table_descriptive(stats: dict):
    """
    Creates an HTML table for descriptive stats (dict of floats).

    Parameters
    ----------
    stats : dict
        A dictionary of descriptive stats, e.g.:
        {
            'price_mean': 123.45,
            'price_std': 67.89,
            'price_min': ...,
            'price_max': ...,
            'logret_mean': ...,
            ...
        }

    Returns
    -------
    dash.html.Table
        A Dash HTML table displaying the metrics.
    """
    rows = []
    for k, v in stats.items():
        row = html.Tr([
            html.Td(k),
            html.Td(f"{v:.5f}")  # round to 5 decimal places
        ])
        rows.append(row)

    return html.Table([
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
        html.Tbody(rows)
    ])


def create_table_forecast(forecast_df: pd.DataFrame):
    """
    Creates an HTML table for the forecasted prices.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Must contain ['date', 'forecast_price'].

    Returns
    -------
    dash.html.Table
        A Dash HTML table displaying each date and its forecasted price.
    """
    rows = []
    for i, row in forecast_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        price_str = f"{row['forecast_price']:.2f}"
        rows.append(
            html.Tr([
                html.Td(date_str),
                html.Td(price_str)
            ])
        )

    return html.Table([
        html.Thead(html.Tr([html.Th("Date"), html.Th("Forecasted Price")], style={'fontWeight':'bold'})),
        html.Tbody(rows)
    ])