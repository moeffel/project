"""
plots.py

This module provides functions to generate various Plotly charts and HTML tables for financial data analysis. The functionalities include:

1. price_plot: Plots actual and forecasted prices.
2. histogram_plot: Plots the distribution of log returns with KDE and normal distribution overlays.
3. qq_plot: Generates a Q-Q plot for log returns.
4. acf_plot: Plots the Autocorrelation Function (ACF) for log returns.
5. pacf_plot: Plots the Partial Autocorrelation Function (PACF) for log returns.
6. create_table_descriptive: Creates an HTML table summarizing descriptive statistics.
7. create_table_forecast: Creates an HTML table for forecasted prices.
8. residual_plot: Visualizes model residuals.

Each function includes comprehensive docstrings, doctests where applicable, and comments to ensure clarity and ease of understanding.

Author: OpenAI ChatGPT
Date: 2025-01-26
"""

import plotly.graph_objs as go
from dash import html
import pandas as pd
import numpy as np
import io
import base64
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm

# 1. Price Plot
def price_plot(df: pd.DataFrame, forecast_df: pd.DataFrame = None, mode: str = 'backtest') -> go.Figure:
    """
    Generates a Plotly line chart showing actual prices and optionally forecasted prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing historical price data with at least 'date' and 'price' columns.
    forecast_df : pd.DataFrame, optional
        DataFrame containing forecasted price data with at least 'date' and 'forecast_price' columns.
        Defaults to None.
    mode : str, optional
        Mode of the plot, e.g., 'backtest'. Currently unused but reserved for future extensions.
        Defaults to 'backtest'.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the price plot.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    ...     'price': np.random.lognormal(mean=0, sigma=0.1, size=100).cumsum()
    ... })
    >>> fig = price_plot(df)
    >>> fig.show()
    """
    # Initialize a new Plotly figure
    fig = go.Figure()
    
    # Add actual price trace
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
    ))
    
    # If forecast data is provided and not empty, add forecast price trace
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast_price'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dot', width=2),
            marker=dict(size=6),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: $%{y:,.2f}<extra></extra>"
        ))
    
    # Update layout with titles and styling
    fig.update_layout(
        title='Price + Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# 2. Histogram Plot
def histogram_plot(df: pd.DataFrame, dist_type: str = 'normal') -> go.Figure:
    """
    Generates a Plotly histogram of log returns with a kernel density estimate (KDE)
    and a normal distribution overlay.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'log_return' column.
    dist_type : str, optional
        The type of distribution assumed for overlay. Defaults to 'normal'.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the histogram with KDE and normal distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'log_return': np.random.normal(loc=0, scale=0.01, size=1000)
    ... })
    >>> fig = histogram_plot(df)
    >>> fig.show()
    """
    # Extract log returns and compute statistics
    log_returns = df['log_return'].dropna()
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    # Initialize a new Plotly figure
    fig = go.Figure()
    
    # Add histogram of log returns
    fig.add_trace(go.Histogram(
        x=log_returns,
        histnorm='probability density',
        name='Log Returns',
        marker_color='#636EFA'
    ))

    # Compute Kernel Density Estimate (KDE) using statsmodels
    kde = sm.nonparametric.KDEUnivariate(log_returns)
    kde.fit(bw=std_return/2)  # Bandwidth set to half the standard deviation
    fig.add_trace(go.Scatter(
        x=kde.support,
        y=kde.density,
        mode='lines',
        name='Kernel Density',
        line=dict(color='#FF7F0E', width=2.5),
    ))

    # Overlay normal distribution curve
    x_norm = np.linspace(log_returns.min(), log_returns.max(), 500)
    y_norm = norm.pdf(x_norm, mean_return, std_return)
    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Dist',
        line=dict(color='#2CA02C', dash='dot', width=2)
    ))

    # Update layout with titles and styling
    fig.update_layout(
        title=f'Return Distribution (Assumed: {dist_type})',
        xaxis_title='Log Returns',
        yaxis_title='Density',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# 3. QQ Plot
def qq_plot(df: pd.DataFrame) -> go.Figure:
    """
    Generates a Q-Q plot for log returns using Plotly by leveraging matplotlib
    and embedding the generated image.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'log_return' column.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the Q-Q plot as an embedded image.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'log_return': np.random.normal(loc=0, scale=0.01, size=1000)
    ... })
    >>> fig = qq_plot(df)
    >>> fig.show()
    """
    # Extract log returns
    log_returns = df['log_return'].dropna()
    
    # Create a matplotlib figure for the Q-Q plot
    plt.figure(figsize=(8, 6))
    _ = sm.qqplot(log_returns, line='45', fit=True)
    plt.title('Q-Q Plot')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Encode the image in base64 to embed in Plotly
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create a Plotly figure and add the image as a layout image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1,
            xref="paper", yref="paper",
            sizex=1, sizey=1,
            layer="below"
        )
    )
    
    # Update layout to adjust sizing and titles
    fig.update_layout(
        width=700, height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
        template='plotly_white',
        title='Q-Q Plot of Log Returns'
    )
    
    return fig

# 4. ACF Plot
def acf_plot(df: pd.DataFrame, lags: int = 40) -> go.Figure:
    """
    Generates an Autocorrelation Function (ACF) plot for log returns using Plotly by leveraging matplotlib
    and embedding the generated image.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'log_return' column.
    lags : int, optional
        The number of lags to display in the ACF plot. Defaults to 40.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the ACF plot as an embedded image.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'log_return': np.random.normal(loc=0, scale=0.01, size=1000)
    ... })
    >>> fig = acf_plot(df, lags=30)
    >>> fig.show()
    """
    # Create a matplotlib figure for the ACF plot
    plt.figure(figsize=(10, 4))
    plot_acf(df['log_return'].dropna(), lags=lags, alpha=0.05)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Encode the image in base64 to embed in Plotly
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create a Plotly figure and add the image as a layout image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1,
            xref="paper", yref="paper",
            sizex=1, sizey=1,
            layer="below"
        )
    )
    
    # Update layout to adjust sizing and titles
    fig.update_layout(
        width=700, height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
        template='plotly_white',
        title='Autocorrelation Function (ACF)'
    )
    
    return fig

# 5. PACF Plot
def pacf_plot(df: pd.DataFrame, lags: int = 40) -> go.Figure:
    """
    Generates a Partial Autocorrelation Function (PACF) plot for log returns using Plotly by leveraging matplotlib
    and embedding the generated image.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'log_return' column.
    lags : int, optional
        The number of lags to display in the PACF plot. Defaults to 40.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the PACF plot as an embedded image.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'log_return': np.random.normal(loc=0, scale=0.01, size=1000)
    ... })
    >>> fig = pacf_plot(df, lags=30)
    >>> fig.show()
    """
    # Create a matplotlib figure for the PACF plot
    plt.figure(figsize=(10, 4))
    plot_pacf(df['log_return'].dropna(), lags=lags, alpha=0.05, method='yw')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Encode the image in base64 to embed in Plotly
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create a Plotly figure and add the image as a layout image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1,
            xref="paper", yref="paper",
            sizex=1, sizey=1,
            layer="below"
        )
    )
    
    # Update layout to adjust sizing and titles
    fig.update_layout(
        width=700, height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
        template='plotly_white',
        title='Partial Autocorrelation Function (PACF)'
    )
    
    return fig

# 6. Descriptive Table
def create_table_descriptive(stats: dict) -> html.Table:
    """
    Creates an HTML table summarizing descriptive statistics.

    Parameters
    ----------
    stats : dict
        A dictionary containing statistical metrics with keys matching the metric_config.

    Returns
    -------
    html.Table
        A Dash HTML Table component displaying the metrics and their values.

    Examples
    --------
    >>> stats = {
    ...     'price_mean': 150.25,
    ...     'price_std': 15.67,
    ...     'logret_mean': 0.0012,
    ...     'mae': 0.0123
    ... }
    >>> table = create_table_descriptive(stats)
    >>> table
    <Table object with metrics>
    """
    # Configuration mapping internal keys to display labels
    metric_config = {
        'price_mean': "Price Mean",
        'price_std': "Price Std Dev",
        'price_min': "Price Min",
        'price_max': "Price Max",
        'price_skew': "Price Skewness",
        'price_kurtosis': "Price Kurtosis",
        'logret_mean': "Mean Log Return",
        'logret_std': "Std Dev Log Return",
        'logret_min': "Min Log Return",
        'logret_max': "Max Log Return",
        'logret_skew': "Log Return Skewness",
        'logret_kurtosis': "Log Return Kurtosis",
        'model_aic': "Model AIC",
        'model_bic': "Model BIC",
        'mae': "MAE",
        'rmse': "RMSE",
        'mape': "MAPE",
    }

    rows = []
    for key, label in metric_config.items():
        if key in stats:
            value = stats[key]
            # Format float values for better readability
            if isinstance(value, float):
                if abs(value) < 1000:
                    value_str = f"{value:.4f}"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            # Append a table row with metric label and value
            rows.append(html.Tr([
                html.Td(label, style={'fontWeight': 'bold', 'padding': '8px'}),
                html.Td(value_str, style={'padding': '8px'})
            ]))

    # Create the HTML table with headers and rows
    table = html.Table(
        [
            html.Thead(
                html.Tr([html.Th("Metric", style={'padding': '8px'}), html.Th("Value", style={'padding': '8px'})])
            ),
            html.Tbody(rows)
        ],
        style={
            'borderCollapse': 'collapse',
            'width': '100%',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        }
    )
    
    return table

# 7. Forecast Table
def create_table_forecast(forecast_df: pd.DataFrame) -> html.Table:
    """
    Creates an HTML table displaying forecasted prices along with changes from the previous forecast.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing forecasted prices with at least 'date' and 'forecast_price' columns.

    Returns
    -------
    html.Table
        A Dash HTML Table component displaying the forecasted prices and their changes.

    Examples
    --------
    >>> forecast_df = pd.DataFrame({
    ...     'date': pd.date_range(start='2025-01-01', periods=5, freq='D'),
    ...     'forecast_price': [150.25, 151.30, 150.80, 152.00, 151.50]
    ... })
    >>> table = create_table_forecast(forecast_df)
    >>> table
    <Table object with forecasted prices>
    """
    rows = []
    prev_price = None  # To store the previous forecast price for comparison

    for _, row in forecast_df.iterrows():
        # Format the date
        date = row['date']
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Format the forecast price
        price = row['forecast_price']
        price_str = f"${price:,.2f}"
        
        # Calculate change and percentage change from previous price
        if prev_price is not None:
            change = price - prev_price
            pct_change = (change / prev_price) * 100 if prev_price != 0 else 0
            # Determine the arrow and color based on the change
            if change > 0:
                arrow = '▲'
                color = '#2ca02c'  # Green for increase
            elif change < 0:
                arrow = '▼'
                color = '#d62728'  # Red for decrease
            else:
                arrow = ''
                color = 'inherit'  # No change
            change_str = f"{arrow} {abs(pct_change):.2f}%"
        else:
            change_str = ""
            color = 'inherit'
        
        # Append a table row with date, forecast price, and change
        rows.append(html.Tr([
            html.Td(date_str, style={'padding': '8px'}),
            html.Td(price_str, style={'padding': '8px'}),
            html.Td(change_str, style={'color': color, 'fontWeight': 'bold', 'padding': '8px'})
        ]))
        
        # Update previous price for next iteration
        prev_price = price

    # Create the HTML table with headers and rows
    table = html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Date", style={'padding': '8px'}),
                    html.Th("Forecast", style={'padding': '8px'}),
                    html.Th("Change", style={'padding': '8px'})
                ])
            ),
            html.Tbody(rows)
        ],
        style={
            'borderCollapse': 'collapse',
            'width': '100%',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        }
    )
    
    return table

# 8. Residual Plot
def residual_plot(residuals: pd.Series, title: str = "Standardized Residuals") -> go.Figure:
    """
    Creates a Plotly line plot of model residuals with a horizontal line at zero for reference.

    Parameters
    ----------
    residuals : pd.Series
        The residuals from the model. The index can be dates or numerical indices.
    title : str, optional
        The title of the plot. Defaults to "Standardized Residuals".

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the residuals plot.

    Examples
    --------
    >>> import pandas as pd
    >>> residuals = pd.Series(np.random.normal(loc=0, scale=1, size=100))
    >>> fig = residual_plot(residuals, title="Model Residuals")
    >>> fig.show()
    """
    # Determine the x-axis values based on the residuals index
    if 'date' in residuals.index.names:
        x_values = residuals.index.get_level_values('date')
    else:
        x_values = residuals.index if isinstance(residuals.index, pd.DatetimeIndex) else np.arange(len(residuals))

    # Initialize a new Plotly figure
    fig = go.Figure()
    
    # Add residuals trace
    fig.add_trace(go.Scatter(
        x=x_values,
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='royalblue', width=2)
    ))
    
    # Determine the x-axis range for the horizontal line at y=0
    if isinstance(x_values, pd.DatetimeIndex):
        x_start = x_values.min()
        x_end = x_values.max()
    elif isinstance(x_values, (np.ndarray, list, pd.Index)):
        if np.issubdtype(x_values.dtype, np.datetime64):
            x_start = pd.to_datetime(x_values.min())
            x_end = pd.to_datetime(x_values.max())
        else:
            x_start = x_values[0]
            x_end = x_values[-1]
    else:
        x_start = x_values[0]
        x_end = x_values[-1]
    
    # Add a horizontal dashed red line at y=0
    fig.add_shape(
        type='line',
        y0=0, y1=0,
        x0=x_start,
        x1=x_end,
        line=dict(color='red', width=2, dash='dash')
    )
    
    # Update layout with titles, labels, and styling
    fig.update_layout(
        title=title,
        xaxis_title="Date" if 'date' in residuals.index.names else "Index",
        yaxis_title="Residual",
        template="plotly_white",
        hovermode='x unified',
        title_font=dict(size=16, color='darkblue'),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        legend=dict(font=dict(size=12)),
        height=350  # Adjust the height as needed
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    return fig