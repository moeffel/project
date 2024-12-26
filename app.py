"""
app.py

Main Dash app that ties together data loading, modeling, 
and visualization. Users can adjust parameters and see 
updated forecasts.

Author: Your Name
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Import from our local modules
from data_loader import fetch_data_coingecko, preprocess_data, train_test_split
from model import fit_arima_garch, forecast_arima_garch
from utils import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from plots import price_plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create Dash app instance
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # For deployment on Heroku (Gunicorn will look for 'server')

# Dictionary mapping user-friendly crypto labels to CoinGecko IDs
cryptos = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Dogecoin (DOGE)': 'dogecoin',
    'Solana (SOL)': 'solana'
}

# Layout: All Dash components here
app.layout = html.Div([
    html.H1("ARIMA-GARCH Cryptocurrency Forecast Dashboard"),
    
    # Cryptocurrency Selection
    html.Div([
        html.Label('Select Cryptocurrency:'),
        dcc.Dropdown(
            id='crypto-dropdown',
            options=[{'label': name, 'value': cid} for name, cid in cryptos.items()],
            value='bitcoin',  # default
            multi=False,
            clearable=False
        )
    ], style={'margin-bottom': '20px'}),

    # ARIMA (p,d,q) Input
    html.Div([
        html.Label("ARIMA (p, d, q):"),
        dcc.Input(id='arima-p', type='number', value=1, placeholder='p', style={'margin-right': '10px'}),
        dcc.Input(id='arima-d', type='number', value=0, placeholder='d', style={'margin-right': '10px'}),
        dcc.Input(id='arima-q', type='number', value=1, placeholder='q', style={'margin-right': '10px'}),
    ], style={'margin-bottom': '20px'}),

    # GARCH (p,q) Input
    html.Div([
        html.Label("GARCH (p, q):"),
        dcc.Input(id='garch-p', type='number', value=1, placeholder='p', style={'margin-right': '10px'}),
        dcc.Input(id='garch-q', type='number', value=1, placeholder='q', style={'margin-right': '10px'})
    ], style={'margin-bottom': '20px'}),

    # Distribution Selection
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
            clearable=False
        )
    ], style={'margin-bottom': '20px', 'width': '250px'}),

    # Forecast horizon
    html.Div([
        html.Label("Forecast Horizon (days):"),
        dcc.Input(id='forecast-horizon', type='number', value=30, style={'margin-right': '10px'})
    ], style={'margin-bottom': '20px'}),

    # Buttons
    html.Div([
        html.Button("Refresh Data", id='refresh-data', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Run Model", id='run-model', n_clicks=0, style={'margin-right': '10px'})
    ], style={'margin-bottom': '20px'}),

    # Loading Spinner
    dcc.Loading(
        id="loading-icon",
        children=[html.Div(id='status')],
        type="default"
    ),

    # Graph output
    dcc.Graph(id='price-plot'),

    # Performance Metrics
    html.H2("Performance Metrics (Test Set)"),
    html.Div(id='performance-metrics'),
])

@app.callback(
    # We want to update these 3 items in the UI:
    Output('status', 'children'),
    Output('price-plot', 'figure'),
    Output('performance-metrics', 'children'),
    # Triggers for the callback: button clicks
    Input('run-model', 'n_clicks'),
    Input('refresh-data', 'n_clicks'),
    # States: we read these values from the UI but they don't individually trigger the callback
    State('crypto-dropdown', 'value'),
    State('arima-p', 'value'),
    State('arima-d', 'value'),
    State('arima-q', 'value'),
    State('garch-p', 'value'),
    State('garch-q', 'value'),
    State('dist-dropdown', 'value'),
    State('forecast-horizon', 'value'),
)
def update_forecast(run_clicks, refresh_clicks, coin_id, p, d, q, gp, gq, dist, horizon):
    """
    Fetches fresh data (if 'Refresh Data' clicked), 
    trains an ARIMA-GARCH model with the chosen parameters,
    forecasts prices, and displays performance metrics.

    Parameters
    ----------
    run_clicks : int
        Number of times "Run Model" is clicked (Dash automatically manages this).
    refresh_clicks : int
        Number of times "Refresh Data" is clicked (not used much here, but can be used for logic).
    coin_id : str
        CoinGecko ID (e.g., 'bitcoin').
    p, d, q : int
        ARIMA hyperparameters.
    gp, gq : int
        GARCH hyperparameters.
    dist : str
        Residual distribution for GARCH ('normal', 't', 'skewt').
    horizon : int
        Number of days to forecast.

    Returns
    -------
    (str, plotly.graph_objs.Figure, str)
        - Status message
        - Figure object for the price forecast
        - Performance metrics text
    """
    # 1) Load data from CoinGecko for the selected coin
    #    Here we fetch 2 years (730 days) to have enough historical data
    df = fetch_data_coingecko(coin_id, days=730)

    # 2) Preprocess data (sort, dropna, compute log returns)
    df = preprocess_data(df)

    # 3) Split data into train/test sets (80%/20%)
    train_df, test_df = train_test_split(df, train_ratio=0.8)

    # 4) Fit ARIMA-GARCH model on the training set's log returns
    arima_model, garch_res = fit_arima_garch(
        train_returns=train_df['log_return'],
        arima_order=(p, d, q),
        garch_order=(gp, gq),
        dist=dist
    )

    # 5) Forecast 'horizon' days of log returns
    forecast_df = forecast_arima_garch(
        arima_model, 
        garch_res, 
        steps=horizon
    )

    # 6) Convert forecasted log returns to forecasted prices
    #    Approach:
    #    - Start from the last known price in the training+testing dataset
    #    - Use the predicted mean_return sequentially.
    last_price = df['price'].iloc[-1]  # last known real price
    # Cumulative product of (1 + log_return) is an approximation for price ratio,
    # but to be precise: predicted_price[t+1] = predicted_price[t] * exp(log_return[t+1])
    # We stored 'mean_return' in normal space, so we can do:
    # price[t+1] = price[t] * exp(mean_return[t+1])
    # We'll approximate with (1 + mean_return) if returns are small, or do an explicit exp if needed:
    # Just for demonstration, let's do exp(mean_return):
    cumulative_factor = np.exp(forecast_df['mean_return']).cumprod()
    forecast_prices = last_price * cumulative_factor

    # Create a new date range for the forecast period
    forecast_dates = pd.date_range(start=df['date'].iloc[-1], periods=horizon+1, freq='D')[1:]  
    # we skip the first day because it's the same as the last known date

    # Merge these into a new forecast DataFrame
    forecast_output_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast_price': forecast_prices.values
    })

    # 7) Calculate Performance Metrics on the test set
    #    - We'll take the first 'horizon' points of the test set for comparison if available
    test_slice = test_df.iloc[:horizon].copy()
    # Merge them on the date to compare actual vs forecast
    merged_df = pd.merge(
        test_slice[['date','price']], 
        forecast_output_df[['date','forecast_price']], 
        on='date', 
        how='inner'
    )

    # If merged is empty, it might mean our test set is shorter than the horizon
    if len(merged_df) > 0:
        mae_val = mean_absolute_error(merged_df['price'], merged_df['forecast_price'])
        rmse_val = root_mean_squared_error(merged_df['price'], merged_df['forecast_price'])
        mape_val = mean_absolute_percentage_error(merged_df['price'], merged_df['forecast_price'])
    else:
        mae_val, rmse_val, mape_val = (None, None, None)

    # 8) Generate the figure for actual vs. forecasted price
    fig = price_forecast_plot(df, forecast_output_df)

    # 9) Prepare status and performance messages
    status_msg = "Model run complete. Data refreshed or re-run with new parameters."
    if mae_val is not None:
        performance_msg = (
            f"MAE: {mae_val:.4f} | "
            f"RMSE: {rmse_val:.4f} | "
            f"MAPE: {mape_val:.2%}"
        )
    else:
        performance_msg = "Not enough overlap between test set and forecast dates for metrics."

    return status_msg, fig, performance_msg


if __name__ == "__main__":
    # For local development, run the server
    app.run_server(debug=True)