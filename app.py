# app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Local modules
from data_loader import fetch_data_yahoo, preprocess_data, train_test_split, compute_descriptive_stats
from model import fit_arima_garch, forecast_arima_garch, auto_tune_arima_garch
from utils import (mean_absolute_error, root_mean_squared_error, 
                  mean_absolute_percentage_error, adf_test)
from plots import (price_plot, histogram_plot, qq_plot, 
                  acf_plot, pacf_plot, create_table_descriptive, 
                  create_table_forecast)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Crypto mapping
CRYPTO_MAP = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Dogecoin (DOGE)': 'dogecoin',
    'Solana (SOL)': 'solana'
}

app.layout = html.Div([
    html.H1("ARIMA-GARCH Crypto Forecasting Dashboard", style={'textAlign': 'center'}),
    
    # Controls Section
    html.Div([
        # New Parameter Mode Selector
        html.Div([
            html.Label("Parameter Mode:"),
            dcc.RadioItems(
                id='param-mode',
                options=[
                    {'label': ' Manual', 'value': 'manual'},
                    {'label': ' Auto-Tune', 'value': 'auto'}
                ],
                value='manual',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
            )
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=datetime(2015, 1, 1),
                max_date_allowed=datetime.today(),
                start_date=datetime(2021, 1, 1),
                end_date=datetime.today(),
                display_format='YYYY-MM-DD'
            )
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '20px'}),

        html.Div([
            html.Label("Select Cryptocurrency:"),
            dcc.Dropdown(
                id='crypto-dropdown',
                options=[{'label': k, 'value': v} for k, v in CRYPTO_MAP.items()],
                value='bitcoin',
                clearable=False
            )
        ], style={'width': '20%', 'display': 'inline-block'}),
        
        # Modified ARIMA/GARCH Inputs
        html.Div([
            html.Label("ARIMA Parameters:"),
            html.Div([
                dcc.Input(id='arima-p', type='number', value=1, min=0, 
                         style={'width': '60px', 'marginRight': '10px'}),
                dcc.Input(id='arima-d', type='number', value=0, min=0,
                         style={'width': '60px', 'marginRight': '10px'}),
                dcc.Input(id='arima-q', type='number', value=1, min=0,
                         style={'width': '60px'})
            ])
        ], id='arima-controls', style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px'}),
        
        html.Div([
            html.Label("GARCH Parameters:"),
            html.Div([
                dcc.Input(id='garch-p', type='number', value=1, min=0,
                         style={'width': '60px', 'marginRight': '10px'}),
                dcc.Input(id='garch-q', type='number', value=1, min=0,
                         style={'width': '60px'})
            ])
        ], id='garch-controls', style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px'}),
        
        html.Div([
            html.Label("Forecast Days:"),
            dcc.Input(id='forecast-horizon', type='number', value=30, min=1,
                     style={'width': '100px'})
        ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '20px'}),
    ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),
    
    # Action Buttons
    html.Div([
        html.Button("Run Analysis", id='run-button', n_clicks=0,
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 
                         'padding': '10px 20px', 'borderRadius': '5px'}),
        html.Div(id='status-message', style={'color': 'red', 'marginLeft': '20px'})
    ], style={'padding': '20px'}),
    
    # Main Content (unchanged)
    html.Div([
        html.Div([
            dcc.Graph(id='price-plot', style={'height': '400px'}),
            html.Div([
                html.Div([dcc.Graph(id='hist-plot')], 
                        style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='qq-plot')], 
                        style={'width': '50%', 'display': 'inline-block'})
            ]),
            html.Div([
                html.Div([dcc.Graph(id='acf-plot')], 
                        style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='pacf-plot')], 
                        style={'width': '50%', 'display': 'inline-block'})
            ])
        ], style={'width': '70%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Performance Metrics"),
            html.Div(id='stats-table', style={'marginBottom': '20px'}),
            html.H3("Forecast Prices"),
            html.Div(id='forecast-table')
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 
                 'padding': '20px', 'backgroundColor': '#f9f9f9'})
    ])
])

# New callback to disable inputs in auto mode
@app.callback(
    [Output('arima-p', 'disabled'),
     Output('arima-d', 'disabled'),
     Output('arima-q', 'disabled'),
     Output('garch-p', 'disabled'),
     Output('garch-q', 'disabled')],
    [Input('param-mode', 'value')]
)
def toggle_inputs(mode):
    disabled = mode == 'auto'
    return [disabled, disabled, disabled, disabled, disabled]

# Modified main callback
@app.callback(
    [Output('status-message', 'children'),
     Output('price-plot', 'figure'),
     Output('hist-plot', 'figure'),
     Output('qq-plot', 'figure'),
     Output('acf-plot', 'figure'),
     Output('pacf-plot', 'figure'),
     Output('stats-table', 'children'),
     Output('forecast-table', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('crypto-dropdown', 'value'),
     State('param-mode', 'value'),
     State('arima-p', 'value'),
     State('arima-d', 'value'),
     State('arima-q', 'value'),
     State('garch-p', 'value'),
     State('garch-q', 'value'),
     State('forecast-horizon', 'value')]
)
def update_all_components(n_clicks, start_date, end_date, coin_id, param_mode, p, d, q, garch_p, garch_q, horizon):
    start_date = start_date.split('T')[0] if start_date else None
    end_date = end_date.split('T')[0] if end_date else None

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    try:
        # 1. Data Loading & Preprocessing
        raw_df = fetch_data_yahoo(coin_id, start=start_date, end=end_date)
        processed_df = preprocess_data(raw_df)
        
        if len(processed_df) < 30:
            raise ValueError("Insufficient data (minimum 30 days required)")
            
        # 2. Train-Test Split (modified for time-based split)
        test_size = horizon
        split_index = len(processed_df) - test_size
        if split_index < 0:
            raise ValueError(f"Need at least {horizon} days of historical data for forecasting")
            
        train_df = processed_df.iloc[:split_index]
        test_df = processed_df.iloc[split_index:]

        # 3. Parameter Selection
        if param_mode == 'auto':
            # Auto-tuning logic
            best_params = auto_tune_arima_garch(train_df['log_return'])
            p, d, q = best_params.get('arima', (1,0,1))
            garch_p, garch_q = best_params.get('garch', (1,1))
            param_status = f"Auto-selected: ARIMA({p},{d},{q}) GARCH({garch_p},{garch_q})"
        else:
            param_status = f"Manual: ARIMA({p},{d},{q}) GARCH({garch_p},{garch_q})"

        # 4. Model Fitting
        arima_model, garch_model, scale = fit_arima_garch(
            train_df['log_return'],
            arima_order=(p, d, q),
            garch_order=(garch_p, garch_q),
            rescale_data=True
        )
        
        # 5. Forecasting (using test period dates)
        forecast = forecast_arima_garch(arima_model, garch_model, horizon, scale)
        forecast_dates = test_df['date'].values
        
        # 6. Create Forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': train_df['price'].iloc[-1] * np.exp(forecast['mean_return']).cumprod().values
        })
        
        # 7. Calculate Metrics
        merged = test_df.merge(forecast_df, on='date')
        metrics = {
            'forecast_mae': mean_absolute_error(merged['price'], merged['forecast_price']),
            'forecast_rmse': root_mean_squared_error(merged['price'], merged['forecast_price']),
            'forecast_mape': mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
        }
        status = f"{param_status} | MAE: {metrics['forecast_mae']:.4f}, RMSE: {metrics['forecast_rmse']:.4f}, MAPE: {metrics['forecast_mape']:.2%}"
        
        # 8. Prepare Statistics
        model_metrics = {
            'model_aic': arima_model.aic + garch_model.aic,
            'model_bic': arima_model.bic + garch_model.bic
        }
        
        full_stats = {
            **compute_descriptive_stats(processed_df),
            **model_metrics,
            **metrics
        }
        
        # 9. Generate Components
        return (
            status,
            price_plot(processed_df, forecast_df),
            histogram_plot(processed_df),
            qq_plot(processed_df),
            acf_plot(processed_df),
            pacf_plot(processed_df),
            create_table_descriptive(full_stats),
            create_table_forecast(forecast_df)
        )
    
    except ValueError as ve:
        return f"Validation Error: {str(ve)}", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []
    except Exception as e:
        return f"Unexpected Error: {str(e)}", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)