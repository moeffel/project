# app.py
# Main Dash app that ties together data loading (Yahoo Finance),
# modeling (ARIMA-GARCH), extensive EDA visualizations, 
# and final forecast tables.
# Author: Your Name

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Local modules
from data_loader import (
    fetch_data_yahoo,
    preprocess_data,
    train_test_split,
    compute_descriptive_stats
)
# Note: fit_arima_garch now returns 3 items, including scale_factor
from model import fit_arima_garch, forecast_arima_garch
from utils import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    adf_test
)
from plots import (
    price_plot,
    histogram_plot,
    qq_plot,
    acf_plot,
    pacf_plot,
    create_table_descriptive,
    create_table_forecast
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Mapping for user selection => coin_id
cryptos = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Dogecoin (DOGE)': 'dogecoin',
    'Solana (SOL)': 'solana'
}

app.layout = html.Div([
    html.H1("ARIMA-GARCH Crypto Forecast Dashboard (Yahoo Finance)"),

    # Coin Selection
    html.Div([
        html.Label("Select Cryptocurrency:"),
        dcc.Dropdown(
            id='crypto-dropdown',
            options=[{'label': k, 'value': v} for k, v in cryptos.items()],
            value='bitcoin',
            multi=False,
            clearable=False,
            style={'width': '300px'}
        )
    ], style={'margin-bottom': '20px'}),

    # ARIMA (p, d, q)
    html.Div([
        html.Label("ARIMA (p, d, q):"),
        dcc.Input(id='arima-p', type='number', value=1, style={'margin-right': '6px'}),
        dcc.Input(id='arima-d', type='number', value=0, style={'margin-right': '6px'}),
        dcc.Input(id='arima-q', type='number', value=1, style={'margin-right': '6px'}),
    ], style={'margin-bottom': '20px'}),

    # GARCH (p, q)
    html.Div([
        html.Label("GARCH (p, q):"),
        dcc.Input(id='garch-p', type='number', value=1, style={'margin-right': '6px'}),
        dcc.Input(id='garch-q', type='number', value=1, style={'margin-right': '6px'}),
    ], style={'margin-bottom': '20px'}),

    # Distribution
    html.Div([
        html.Label("Error Distribution:"),
        dcc.Dropdown(
            id='dist-dropdown',
            options=[
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Student t', 'value': 't'},
                {'label': 'Skewed t', 'value': 'skewt'}
            ],
            value='normal',
            clearable=False,
            style={'width': '250px'}
        )
    ], style={'margin-bottom': '20px'}),

    # Forecast horizon
    html.Div([
        html.Label("Forecast Horizon (days):"),
        dcc.Input(id='forecast-horizon', type='number', value=30, style={'margin-right': '6px'})
    ], style={'margin-bottom': '20px'}),

    # Buttons
    html.Div([
        html.Button("Refresh Data", id='refresh-data', n_clicks=0, style={'margin-right': '15px'}),
        html.Button("Run Model", id='run-model', n_clicks=0, style={'margin-right': '15px'}),
    ], style={'margin-bottom': '20px'}),

    # Status
    html.Div(id='status', style={'color': 'red', 'margin-bottom': '20px'}),

    # Main Price-Forecast Plot
    dcc.Graph(id='price-plot'),

    # Additional EDA Plots
    html.H3("Histogram of Log Returns"),
    dcc.Graph(id='hist-plot'),

    html.H3("Q-Q Plot of Log Returns"),
    dcc.Graph(id='qq-plot'),

    html.H3("ACF Plot (Log Returns)"),
    dcc.Graph(id='acf-plot'),

    html.H3("PACF Plot (Log Returns)"),
    dcc.Graph(id='pacf-plot'),

    # Descriptive Stats Table
    html.H3("Descriptive Statistics"),
    html.Div(id='metrics-table'),

    # Forecasted Prices Table
    html.H3("Forecasted Prices"),
    html.Div(id='forecast-table'),
])

@app.callback(
    [
        Output('status', 'children'),
        Output('price-plot', 'figure'),
        Output('hist-plot', 'figure'),
        Output('qq-plot', 'figure'),
        Output('acf-plot', 'figure'),
        Output('pacf-plot', 'figure'),
        Output('metrics-table', 'children'),
        Output('forecast-table', 'children'),
    ],
    [
        Input('run-model', 'n_clicks'),
        Input('refresh-data', 'n_clicks')
    ],
    [
        State('crypto-dropdown', 'value'),
        State('arima-p', 'value'),
        State('arima-d', 'value'),
        State('arima-q', 'value'),
        State('garch-p', 'value'),
        State('garch-q', 'value'),
        State('dist-dropdown', 'value'),
        State('forecast-horizon', 'value'),
    ]
)
def update_forecast(run_clicks, refresh_clicks,
                    coin_id, p, d, q, gp, gq, dist, horizon):
    """
    Callback that runs the entire pipeline:
    1) Load data (Yahoo)
    2) Preprocess (log returns)
    3) Descriptive stats
    4) Train/Test split
    5) Fit ARIMA-GARCH
    6) Forecast log returns & convert to price
    7) Create EDA plots
    8) Return figure & tables
    """
    try:
        # 1) Load data for the chosen coin
        df = fetch_data_yahoo(coin_id, start="2019-01-01", end=None)

        # 2) Preprocess
        df = preprocess_data(df)
        if len(df) < 10:
            raise ValueError("Not enough data points after preprocess. Need >= 10 rows for ARIMA-GARCH demo.")

        # 3) Descriptive Stats
        stats_dict = compute_descriptive_stats(df)

        # 4) Train/Test split
        train_df, test_df = train_test_split(df, train_ratio=0.8)
        if len(train_df) < 5:
            raise ValueError("Training set too small after split.")

        # 5) Fit ARIMA-GARCH
        # -- FIX: Unpack 3 values instead of 2 --
        arima_model, garch_res, used_scale = fit_arima_garch(
            train_returns=train_df['log_return'],
            arima_order=(p, d, q),
            garch_order=(gp, gq),
            dist=dist,
            rescale_data=True,    # or False if you want
            scale_factor=1000.0   # adjust if needed
        )

        # 6) Forecast
        # -- FIX: pass used_scale to forecast_arima_garch --
        forecast_df = forecast_arima_garch(
            arima_model,
            garch_res,
            steps=horizon,
            scale_factor=used_scale
        )

        # Convert log returns -> price
        last_price = df['price'].iloc[-1]
        cum_factor = np.exp(forecast_df['mean_return']).cumprod()
        forecast_prices = last_price * cum_factor

        # Create new date range for forecast
        forecast_dates = pd.date_range(
            start=df['date'].iloc[-1],
            periods=horizon+1,
            freq='D'
        )[1:]
        forecast_output_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': forecast_prices.values
        })

        # Calculate performance on test set
        test_slice = test_df.iloc[:horizon]
        merged = pd.merge(
            test_slice[['date','price']],
            forecast_output_df[['date','forecast_price']],
            on='date', how='inner'
        )
        if len(merged) > 0:
            mae_val = mean_absolute_error(merged['price'], merged['forecast_price'])
            rmse_val = root_mean_squared_error(merged['price'], merged['forecast_price'])
            mape_val = mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
            performance_msg = (
                f"MAE: {mae_val:.4f}, "
                f"RMSE: {rmse_val:.4f}, "
                f"MAPE: {mape_val:.2%}"
            )
        else:
            performance_msg = "Not enough overlap to compute metrics."

        # 7) EDA Plots
        fig_price = price_plot(df, forecast_output_df)
        fig_hist = histogram_plot(df)
        fig_qq = qq_plot(df)
        fig_acf = acf_plot(df)
        fig_pacf = pacf_plot(df)

        # Create tables
        table_stats = create_table_descriptive(stats_dict)
        table_forecast = create_table_forecast(forecast_output_df)

        status_msg = f"Model run complete. {performance_msg}"

        return (
            status_msg,        # status
            fig_price,         # price-plot figure
            fig_hist,          # histogram
            fig_qq,            # qq-plot
            fig_acf,           # acf-plot
            fig_pacf,          # pacf-plot
            table_stats,       # metrics-table
            table_forecast     # forecast-table
        )

    except Exception as e:
        import traceback
        error_text = f"ERROR: {e}\nTraceback:\n{traceback.format_exc()}"
        # Return empty figs & empty tables if error
        empty_fig = go.Figure()
        return (
            error_text,
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            html.Div(),
            html.Div()
        )

if __name__ == "__main__":
    app.run_server(debug=True)