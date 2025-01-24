"""
plots.py

Generate Plotly charts and HTML tables.
"""

import plotly.graph_objs as go
from dash import html
import pandas as pd
import numpy as np
import io
import base64
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm

# 1. Price Plot
def price_plot(df: pd.DataFrame, forecast_df: pd.DataFrame = None, mode='backtest') -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
    ))
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
    fig.update_layout(
        title='Price + Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# 2. Histogram Plot
def histogram_plot(df: pd.DataFrame) -> go.Figure:
    log_returns = df['log_return'].dropna()
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    # Basic histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_returns,
        histnorm='probability density',
        name='Log Returns',
        marker_color='#636EFA'
    ))

    # KDE using statsmodels
    kde = sm.nonparametric.KDEUnivariate(log_returns)
    kde.fit(bw=std_return/2)
    fig.add_trace(go.Scatter(
        x=kde.support,
        y=kde.density,
        mode='lines',
        name='Kernel Density',
        line=dict(color='#FF7F0E', width=2.5),
    ))

    # Normal reference
    x_norm = np.linspace(log_returns.min(), log_returns.max(), 500)
    y_norm = norm.pdf(x_norm, mean_return, std_return)
    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Dist',
        line=dict(color='#2CA02C', dash='dot', width=2)
    ))
    
    fig.update_layout(
        title='Return Distribution',
        xaxis_title='Log Returns',
        yaxis_title='Density',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# 3. QQ Plot
def qq_plot(df: pd.DataFrame) -> go.Figure:
    log_returns = df['log_return'].dropna()
    plt.figure(figsize=(8, 6))
    qq = sm.qqplot(log_returns, line='45', fit=True)
    plt.title('Q-Q Plot')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1, xref="paper", yref="paper",
            sizex=1, sizey=1, layer="below"
        )
    )
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
    plt.figure(figsize=(10, 4))
    plot_acf(df['log_return'].dropna(), lags=lags, alpha=0.05)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1, xref="paper", yref="paper",
            sizex=1, sizey=1, layer="below"
        )
    )
    fig.update_layout(
        width=700, height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
        template='plotly_white',
        title='ACF'
    )
    return fig

# 5. PACF Plot
def pacf_plot(df: pd.DataFrame, lags: int = 40) -> go.Figure:
    plt.figure(figsize=(10, 4))
    plot_pacf(df['log_return'].dropna(), lags=lags, alpha=0.05, method='yw')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image}",
            x=0, y=1, xref="paper", yref="paper",
            sizex=1, sizey=1, layer="below"
        )
    )
    fig.update_layout(
        width=700, height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
        template='plotly_white',
        title='PACF'
    )
    return fig

# 6. Descriptive Table
def create_table_descriptive(stats: dict) -> html.Table:
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
    for k, label in metric_config.items():
        if k in stats:
            rows.append(html.Tr([
                html.Td(label, style={'fontWeight': 'bold'}),
                html.Td(f"{stats[k]:.4f}" if abs(stats[k]) < 1000 else f"{stats[k]:.2f}")
            ]))

    return html.Table([
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
        html.Tbody(rows)
    ], style={'borderCollapse': 'collapse', 'width': '100%'})

# 7. Forecast Table
def create_table_forecast(forecast_df: pd.DataFrame) -> html.Table:
    rows = []
    prev_price = None
    for _, row in forecast_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        price_str = f"${row['forecast_price']:.2f}"
        if prev_price is not None:
            change = row['forecast_price'] - prev_price
            pct_change = change / prev_price * 100 if prev_price != 0 else 0
            arrow = '▲' if change > 0 else ('▼' if change < 0 else '')
            color = '#2ca02c' if change > 0 else '#d62728' if change < 0 else 'inherit'
            change_str = f"{arrow} {abs(pct_change):.2f}%"
        else:
            change_str = ""
            color = 'inherit'
        rows.append(html.Tr([
            html.Td(date_str),
            html.Td(price_str),
            html.Td(change_str, style={'color': color, 'fontWeight': 'bold'})
        ]))
        prev_price = row['forecast_price']

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Date"), html.Th("Forecast"), html.Th("Change")
        ])),
        html.Tbody(rows)
    ], style={'borderCollapse': 'collapse', 'width': '100%'}) 