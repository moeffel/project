import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime

# Local modules
from data_loader import fetch_data_yahoo, preprocess_data
from model import fit_arima_garch, forecast_arima_garch, auto_tune_arima_garch
from utils import (
    mean_absolute_error, 
    root_mean_squared_error, 
    mean_absolute_percentage_error,
    compute_descriptive_stats
)
from plots import (
    price_plot, histogram_plot, qq_plot, 
    acf_plot, pacf_plot, create_table_descriptive, 
    create_table_forecast
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Example crypto mapping
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
        html.Div([
            html.Label("Error Distribution:"),
            dcc.RadioItems(
                id='garch-distribution',
                options=[
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': "Student's t", 'value': 't'},
                    {'label': 'Skewed t', 'value': 'skewt'}
                ],
                value='normal',  # default
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '15px'}),

        html.Div([
            html.Label("Forecast Mode:"),
            dcc.RadioItems(
                id='forecast-mode',
                options=[
                    {'label': ' Backtest', 'value': 'backtest'},
                    {'label': ' Future', 'value': 'future'}
                ],
                value='backtest',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '15px'}),
        
        html.Div([
            html.Label("Parameter Mode:"),
            dcc.RadioItems(
                id='param-mode',
                options=[
                    {'label': ' Manual', 'value': 'manual'},
                    {'label': ' Auto-Tune', 'value': 'auto'}
                ],
                value='manual',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '15px'}),
        
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
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '15px'}),

        html.Div([
            html.Label("Select Cryptocurrency:"),
            dcc.Dropdown(
                id='crypto-dropdown',
                options=[{'label': k, 'value': v} for k, v in CRYPTO_MAP.items()],
                value='bitcoin',
                clearable=False
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),
        
        html.Div([
            html.Label("ARIMA Parameters:"),
            html.Div([
                dcc.Input(
                    id='arima-p', 
                    type='number', 
                    value=1, 
                    min=0,
                    style={'width': '60px', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='arima-d', 
                    type='number', 
                    value=0, 
                    min=0,
                    style={'width': '60px', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='arima-q', 
                    type='number', 
                    value=1, 
                    min=0,
                    style={'width': '60px'}
                )
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("GARCH Parameters:"),
            html.Div([
                dcc.Input(
                    id='garch-p', 
                    type='number', 
                    value=1, 
                    min=0,
                    style={'width': '60px', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='garch-q', 
                    type='number', 
                    value=1, 
                    min=0,
                    style={'width': '60px'}
                )
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Forecast Days:"),
            dcc.Input(
                id='forecast-horizon', 
                type='number', 
                value=30, 
                min=1,
                style={'width': '100px'}
            )
        ], style={'width': '10%', 'display': 'inline-block'}),
    ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),
    
    # Action Section
    html.Div([
        html.Button(
            "Run Analysis", 
            id='run-button', 
            n_clicks=0,
            style={
                'backgroundColor': '#4CAF50', 
                'color': 'white', 
                'padding': '10px 20px', 
                'borderRadius': '5px'
            }
        ),
        html.Div(
            id='status-message', 
            style={'color': 'red', 'marginLeft': '20px'}
        )
    ], style={'padding': '20px'}),
    
    # Main Content
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
        ], style={
            'width': '28%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'padding': '20px', 
            'backgroundColor': '#f9f9f9'
        })
    ])
])

@app.callback(
    [Output('arima-p', 'disabled'),
     Output('arima-d', 'disabled'),
     Output('arima-q', 'disabled'),
     Output('garch-p', 'disabled'),
     Output('garch-q', 'disabled')],
    [Input('param-mode', 'value')]
)
def toggle_inputs(mode):
    # If param-mode is 'auto', we disable manual input fields
    disabled = (mode == 'auto')
    return [disabled, disabled, disabled, disabled, disabled]


@app.callback(
    [Output('status-message', 'children'),
     Output('price-plot', 'figure'),
     Output('hist-plot', 'figure'),
     Output('qq-plot', 'figure'),
     Output('acf-plot', 'figure'),
     Output('pacf-plot', 'figure'),
     Output('stats-table', 'children'),
     Output('forecast-table', 'children')],
    [Input('run-button', 'n_clicks'),
     Input('garch-distribution', 'value')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('crypto-dropdown', 'value'),
     State('param-mode', 'value'),
     State('forecast-mode', 'value'),
     State('arima-p', 'value'),
     State('arima-d', 'value'),
     State('arima-q', 'value'),
     State('garch-p', 'value'),
     State('garch-q', 'value'),
     State('forecast-horizon', 'value')]
)
def update_all_components(n_clicks,
                          garch_dist,
                          start_date,
                          end_date,
                          coin_id,
                          param_mode,
                          forecast_mode,
                          p, d, q,
                          garch_p, garch_q,
                          horizon):
    # If nothing triggered, do nothing
    if not dash.callback_context.triggered:
        return dash.no_update

    try:
        # Convert start/end to 'YYYY-MM-DD'
        if start_date:
            start_date = start_date.split('T')[0]
        if end_date:
            end_date = end_date.split('T')[0]

        # 1. Load & preprocess data
        raw_df = fetch_data_yahoo(coin_id, start=start_date, end=end_date)
        processed_df = preprocess_data(raw_df)

        if len(processed_df) < 30:
            raise ValueError("Insufficient data (minimum 30 days required).")

        # 2. Train-Test split or future mode
        if forecast_mode == 'backtest':
            split_index = len(processed_df) - horizon
            if split_index < 0:
                raise ValueError(f"Need at least {horizon} data points for backtest.")
            train_df = processed_df.iloc[:split_index]
            test_df = processed_df.iloc[split_index:]
            forecast_dates = test_df['date'].values
        else:
            train_df = processed_df
            test_df = pd.DataFrame()
            last_date = pd.to_datetime(train_df['date'].iloc[-1])
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(days=1),
                periods=horizon,
                freq='D'
            )

        # 3. Parameter selection
        if param_mode == 'auto':
            best_params = auto_tune_arima_garch(train_df['log_return'])
            p, d, q = best_params['arima']
            garch_p, garch_q = best_params['garch']
            param_status = f"Auto: ARIMA({p},{d},{q}) GARCH({garch_p},{garch_q})"
        else:
            param_status = f"Manual: ARIMA({p},{d},{q}) GARCH({garch_p},{garch_q})"

        # 4. Fit ARIMA-GARCH
        arima_model, garch_model, scale = fit_arima_garch(
            train_df['log_return'],
            arima_order=(p, d, q),
            garch_order=(garch_p, garch_q),
            dist=garch_dist,     # pass the distribution from the UI
            rescale_data=True
        )

        # 5. Forecast
        forecast = forecast_arima_garch(arima_model, garch_model, horizon, scale)

        # 6. Construct forecasted price from forecasted returns
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': (
                train_df['price'].iloc[-1] *
                np.exp(forecast['mean_return']).cumprod().values
            )
        })

        # 7. Evaluate metrics (if backtest)
        metrics = {}
        if forecast_mode == 'backtest':
            merged = test_df.merge(forecast_df, on='date')
            metrics = {
                'mae': mean_absolute_error(merged['price'], merged['forecast_price']),
                'rmse': root_mean_squared_error(merged['price'], merged['forecast_price']),
                'mape': mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
            }
            metric_text = (
                f"MAE: {metrics['mae']:.2f}, "
                f"RMSE: {metrics['rmse']:.2f}, "
                f"MAPE: {metrics['mape']:.2%}"
            )
        else:
            metric_text = f"Forecast for {horizon} days (no backtest)"

        # 8. Descriptive Stats + Model AIC/BIC
        stats = {
            **compute_descriptive_stats(processed_df),
            'model_aic': arima_model.aic + garch_model.aic,
            'model_bic': arima_model.bic + garch_model.bic,
            **metrics
        }

        # 9. Build Figures/Tables
        return (
            f"{param_status} | {metric_text}",
            price_plot(train_df, forecast_df, forecast_mode),
            histogram_plot(processed_df),
            qq_plot(processed_df),
            acf_plot(processed_df),
            pacf_plot(processed_df),
            create_table_descriptive(stats),
            create_table_forecast(forecast_df)
        )

    except ValueError as ve:
        return (f"Validation Error: {str(ve)}",
                go.Figure(), go.Figure(), go.Figure(),
                go.Figure(), go.Figure(), [], [])
    except Exception as e:
        return (f"Unexpected Error: {str(e)}",
                go.Figure(), go.Figure(), go.Figure(),
                go.Figure(), go.Figure(), [], [])

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)
