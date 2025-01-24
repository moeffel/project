"""
plots.py

Generate enhanced Plotly charts with interpretability features,
plus HTML tables with AIC/BIC metrics.

Author: Your Name
"""

import matplotlib
matplotlib.use("Agg")  # For non-GUI environments

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash import html
import matplotlib.pyplot as plt
import io
import base64
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm

# =============================================================================
# 1. PRICE PLOT WITH FORECAST
# =============================================================================
def price_plot(df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> go.Figure:
    """
    Enhanced price plot with forecast visualization and hover annotations.
    """
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
    ))

    # Forecasted prices
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
        title='Price History and Forecast with 95% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

# =============================================================================
# 2. ENHANCED HISTOGRAM WITH KDE
# =============================================================================
def histogram_plot(df: pd.DataFrame) -> go.Figure:
    """
    Professional histogram with dynamic scaling and improved layout.
    """
    log_returns = df['log_return'].dropna()
    
    # Calculate statistics
    mean_return = log_returns.mean()
    std_return = log_returns.std()
    kurt = log_returns.kurtosis()
    
    # Smart bin calculation with caps
    iqr = log_returns.quantile(0.75) - log_returns.quantile(0.25)
    bin_width = 2 * iqr / (len(log_returns) ** (1/3))
    bin_count = min(int(np.ceil((log_returns.max() - log_returns.min()) / bin_width)), 50)
    
    fig = go.Figure()
    
    # ------------------
    # 1. Main Histogram
    # ------------------
    fig.add_trace(go.Histogram(
        x=log_returns,
        xbins=dict(size=bin_width),
        name='Log Returns',
        opacity=0.6,
        marker_color='#636EFA',
        histnorm='probability density',
        hovertemplate="Return: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>"
    ))
    
    # ------------------
    # 2. KDE Plot
    # ------------------
    kde = sm.nonparametric.KDEUnivariate(log_returns)
    kde.fit(bw=std_return/2)  # Adjusted bandwidth
    fig.add_trace(go.Scatter(
        x=kde.support,
        y=kde.density,
        mode='lines',
        name='Kernel Density',
        line=dict(color='#FF7F0E', width=2.5),
        hovertemplate="KDE: %{y:.4f}<extra></extra>"
    ))
    
    # ------------------
    # 3. Normal Reference
    # ------------------
    x_norm = np.linspace(log_returns.min(), log_returns.max(), 500)
    y_norm = norm.pdf(x_norm, mean_return, std_return)
    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#2CA02C', dash='dot', width=2),
        hovertemplate="Normal Ref: %{y:.4f}<extra></extra>"
    ))
    
    # ------------------
    # 4. Annotations & Styling
    # ------------------
    fig.update_layout(
        title=dict(
            text='Return Distribution Analysis',
            x=0.5,
            font=dict(size=20, color='#2F4F4F')
        ),
        xaxis=dict(
            title='Log Return',
            range=[log_returns.min()*1.1, log_returns.max()*1.1],
            gridcolor='#F0F0F0',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='Density',
            range=[0, min(kde.density.max()*1.2, 20)],  # Cap y-axis at 20
            gridcolor='#F0F0F0',
            title_font=dict(size=14)
        ),
        legend=dict(
            x=0.78,
            y=0.95,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#CCCCCC'
        ),
        margin=dict(l=50, r=50, b=80, t=90),
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        annotations=[
            dict(
                x=0.03,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f"Excess Kurtosis: {kurt:.2f}",
                showarrow=False,
                bgcolor='white',
                bordercolor='#666666',
                borderwidth=1,
                font=dict(size=12)
            )
        ]
    )
    
    return fig
# =============================================================================
# 3. PROFESSIONAL Q-Q PLOT WITH CONFIDENCE BANDS
# =============================================================================
def qq_plot(df: pd.DataFrame) -> go.Figure:
    log_returns = df['log_return'].dropna()
    
    # Create matplotlib figure
    plt.figure(figsize=(10, 6), dpi=100)
    
    try:
        # Modern statsmodels with confidence bands
        qq = sm.qqplot(log_returns, line='45', fit=True, confidence_intervals=True)
        plt.title('Q-Q Plot with 95% Confidence Bands')
    except (TypeError, AttributeError):
        # Fallback for older versions
        qq = sm.qqplot(log_returns, line='45', fit=True)
        plt.title('Q-Q Plot')
        
        # Manually add confidence band annotation
        plt.text(0.05, 0.95, 
                 "Confidence bands not available in this statsmodels version",
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    
    # Convert to Plotly-compatible image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create Plotly figure
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
    
    fig.update_layout(
        title='Q-Q Plot of Log Returns',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=800,
        height=600,
        margin=dict(t=60, b=40),
        template='plotly_white'
    )
    
    return fig
# =============================================================================
# 4. ENHANCED DESCRIPTIVE TABLE WITH AIC/BIC
# =============================================================================
def create_table_descriptive(stats: dict) -> html.Table:
    """
    Creates a professional HTML table with metric grouping and formatting.
    """
    metric_config = {
        'price': {
            'mean': ('Mean Price', '.2f'),
            'std': ('Price Volatility', '.2f'),
            'min': ('Minimum Price', '.2f'),
            'max': ('Maximum Price', '.2f')
        },
        'logret': {
            'mean': ('Mean Return', '.4f'),
            'std': ('Return Volatility', '.4f'),
            'skew': ('Return Skewness', '.2f'),
            'kurtosis': ('Return Kurtosis', '.2f')
        },
        'model': {
            'aic': ('AIC', '.1f'),
            'bic': ('BIC', '.1f')
        },
        'forecast': {
            'mae': ('MAE', '.2f'),
            'rmse': ('RMSE', '.2f'),
            'mape': ('MAPE', '.2%')
        }
    }

    rows = []
    for group in ['price', 'logret', 'model', 'forecast']:
        # Add section header
        rows.append(html.Tr([
            html.Td(
                group.title(),
                style={
                    'fontWeight': 'bold',
                    'backgroundColor': '#f8f9fa',
                    'borderTop': '2px solid #dee2e6'
                },
                colSpan=2
            )
        ]))
        
        # Add metrics
        for metric, (label, fmt) in metric_config.get(group, {}).items():
            key = f"{group}_{metric}"
            if key in stats:
                rows.append(html.Tr([
                    html.Td(
                        label,
                        style={
                            'paddingLeft': '20px',
                            'fontWeight': '500'
                        }
                    ),
                    html.Td(
                        f"{stats[key]:{fmt}}",
                    )
                ]))

    return html.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody(rows)
        ],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'margin': '20px 0',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.12)'
        }
    )
# =============================================================================
# 5. FORECAST TABLE WITH TREND INDICATORS
# =============================================================================
def create_table_forecast(forecast_df: pd.DataFrame) -> html.Table:
    """
    Creates an HTML table with trend arrows for forecasted prices.
    """
    rows = []
    prev_price = None
    
    for i, row in forecast_df.iterrows():
        current_price = row['forecast_price']
        date_str = row['date'].strftime('%Y-%m-%d')
        price_str = f"${current_price:,.2f}"
        
        # Trend indicator logic
        if prev_price is not None:
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100
            arrow = '▲' if change > 0 else '▼' if change < 0 else ''
            color = '#2ca02c' if change > 0 else '#d62728' if change < 0 else 'inherit'
            change_str = f"{arrow} {abs(pct_change):.2f}%"
        else:
            change_str = ''
            color = 'inherit'
        
        rows.append(html.Tr([
            html.Td(date_str),
            html.Td(price_str),
            html.Td(
                change_str,
                style={'color': color, 'fontWeight': 'bold'}
            )
        ]))
        
        prev_price = current_price

    return html.Table(
        [
            html.Thead(html.Tr([
                html.Th("Date", style={'width': '40%'}),
                html.Th("Forecast", style={'width': '40%'}),
                html.Th("Change", style={'width': '20%'})
            ])),
            html.Tbody(rows)
        ],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'margin': '20px 0',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.12)'
        }
    )

# =============================================================================
# 6. ENHANCED ACF/PACF PLOTS WITH CONFIDENCE INTERVALS
# =============================================================================
def acf_plot(df: pd.DataFrame, lags: int = 40) -> go.Figure:
    """
    Creates an ACF plot with 95% confidence intervals using Matplotlib/Statsmodels,
    converted to a Plotly-compatible figure with annotations.
    """
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 4)})
    
    # Create Matplotlib figure
    fig_mpl, ax = plt.subplots()
    plot_acf(df['log_return'].dropna(), lags=lags, alpha=0.05, ax=ax, 
             title=f'ACF of Log Returns (Lags 1-{lags})', 
             zero=False, color='#1f77b4')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)
    
    # Style enhancements
    ax.set_facecolor('#f8f9fa')
    fig_mpl.patch.set_facecolor('white')
    plt.tight_layout()
    
    # Convert to Plotly-compatible image
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create Plotly figure with annotations
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
    
    # Add explanatory annotations
    fig.add_annotation(
        x=0.05, y=0.15,
        xref="paper", yref="paper",
        text="Blue area shows 95% confidence interval",
        showarrow=False,
        font=dict(color="#666666", size=12),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(
        title='Autocorrelation Function (ACF)',
        margin=dict(t=40, b=40),
        width=1000,
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white'
    )
    
    return fig

def pacf_plot(df: pd.DataFrame, lags: int = 40) -> go.Figure:
    """
    Creates a PACF plot with 95% confidence intervals using Yule-Walker method,
    converted to Plotly with enhanced styling.
    """
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 4)})
    
    # Create Matplotlib figure
    fig_mpl, ax = plt.subplots()
    plot_pacf(df['log_return'].dropna(), lags=lags, alpha=0.05, ax=ax,
              title=f'PACF of Log Returns (Lags 1-{lags})',
              method='yw', color='#ff7f0e')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Correlation')
    ax.grid(True, alpha=0.3)
    
    # Style enhancements
    ax.set_facecolor('#f8f9fa')
    fig_mpl.patch.set_facecolor('white')
    plt.tight_layout()
    
    # Convert to Plotly-compatible image
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create Plotly figure
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
    
    # Add explanatory annotations
    fig.add_annotation(
        x=0.05, y=0.15,
        xref="paper", yref="paper",
        text="Orange area shows 95% confidence interval",
        showarrow=False,
        font=dict(color="#666666", size=12),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(
        title='Partial Autocorrelation Function (PACF)',
        margin=dict(t=40, b=40),
        width=1000,
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white'
    )
    
    return fig