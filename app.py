import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# local modules
from data_loader import fetch_data_yahoo, preprocess_data
from model import fit_arima_garch, forecast_arima_garch, auto_tune_arima_garch
from utils import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    compute_descriptive_stats,
    adf_test,
    ljung_box_test,
    arch_test
)
from plots import (
    price_plot, histogram_plot, qq_plot, 
    acf_plot, pacf_plot, create_table_descriptive, 
    create_table_forecast, residual_plot
)
import matplotlib.pyplot as plt
import io
import base64
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    
    # ============== CONTROLS SECTION ==============
    html.Div([
        # 1) Distribution
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

        # 2) Forecast Mode
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
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),
        
        # 3) Parameter Mode
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
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),
        
        # 4) Date Range
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

        # 5) Crypto
        html.Div([
            html.Label("Select Cryptocurrency:"),
            dcc.Dropdown(
                id='crypto-dropdown',
                options=[{'label': k, 'value': v} for k, v in CRYPTO_MAP.items()],
                value='bitcoin',
                clearable=False
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),
        
        # 6) ARIMA (p,d,q)
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
        ], style={'width': '15%', 'display': 'inline-block'}),
        
        # 7) GARCH (p,q)
        html.Div([
            html.Label("GARCH Parameters:"),
            html.Div([
                dcc.Input(id='garch-p', type='number', value=1, min=0,
                          style={'width': '60px', 'marginRight': '10px'}),
                dcc.Input(id='garch-q', type='number', value=1, min=0,
                          style={'width': '60px'})
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),
        
        # 8) Forecast Days
        html.Div([
            html.Label("Forecast Days:"),
            dcc.Input(id='forecast-horizon', type='number', value=30, min=1,
                      style={'width': '100px'})
        ], style={'width': '10%', 'display': 'inline-block'}),
    ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),
    
    # ============== ACTION SECTION ==============
    html.Div([
        html.Button("Run Analysis", id='run-button', n_clicks=0,
                    style={'backgroundColor': '#4CAF50', 'color': 'white', 
                           'padding': '10px 20px', 'borderRadius': '5px'}),
        html.Button("Refresh Data", id='refresh-button', n_clicks=0,
                    style={'backgroundColor': '#2196F3', 'color': 'white',
                           'padding': '10px 20px', 'marginLeft':'10px',
                           'borderRadius': '5px'}),
        html.Div(id='status-message', style={'color': 'red', 'marginLeft': '20px'})
    ], style={'padding': '20px'}),

    # ============== MAIN CONTENT ==============
    html.Div([
        # Left side: Plots
        html.Div([
            # Price Plot
            dcc.Graph(id='price-plot', style={'height': '400px'}),
            
            # Hist & QQ
            html.Div([
                html.Div([dcc.Graph(id='hist-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='qq-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # ACF & PACF
            html.Div([
                html.Div([dcc.Graph(id='acf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='pacf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})
            ]),

            # Residual Plot
            dcc.Graph(id='resid-plot', style={'height': '300px'}),

        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right side: Tables & Diagnostics
        html.Div([
            html.H3("Performance Metrics"),
            html.Div(id='stats-table', style={'marginBottom': '20px'}),

            html.H3("Forecast Prices"),
            html.Div(id='forecast-table'),

            html.H3("Diagnostics"),
            html.Div(id='diagnostics-summary', style={'whiteSpace': 'pre-wrap'}),
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 
                  'padding': '20px', 'backgroundColor': '#f9f9f9'})
    ])
])

# ============== CALLBACKS ==============

@app.callback(
    [Output('arima-p', 'disabled'),
     Output('arima-d', 'disabled'),
     Output('arima-q', 'disabled'),
     Output('garch-p', 'disabled'),
     Output('garch-q', 'disabled')],
    [Input('param-mode', 'value')]
)
def toggle_inputs(mode):
    """ Disable ARIMA/GARCH param inputs if param-mode = 'auto'. """
    disabled = (mode == 'auto')
    return [disabled, disabled, disabled, disabled, disabled]


@app.callback(
    [
        Output('status-message', 'children'),
        Output('price-plot', 'figure'),
        Output('hist-plot', 'figure'),
        Output('qq-plot', 'figure'),
        Output('acf-plot', 'figure'),
        Output('pacf-plot', 'figure'),
        Output('resid-plot', 'figure'),
        Output('stats-table', 'children'),
        Output('forecast-table', 'children'),
        Output('diagnostics-summary', 'children')
    ],
    [
        Input('run-button', 'n_clicks'),
        Input('refresh-button', 'n_clicks'),
        Input('garch-distribution', 'value')
    ],
    [
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('crypto-dropdown', 'value'),
        State('param-mode', 'value'),
        State('forecast-mode', 'value'),
        State('arima-p', 'value'),
        State('arima-d', 'value'),
        State('arima-q', 'value'),
        State('garch-p', 'value'),
        State('garch-q', 'value'),
        State('forecast-horizon', 'value')
    ]
)
def update_all_components(run_clicks,
                          refresh_clicks,
                          garch_dist,
                          start_date,
                          end_date,
                          coin_id,
                          param_mode,
                          forecast_mode,
                          p, d, q,
                          garch_p, garch_q,
                          horizon):
    """
    Main callback for:
    - Refreshing data
    - ARIMA-GARCH analysis
    - Generating plots/tables
    - Displaying diagnostics
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        # Convert dates
        if start_date:
            start_date = start_date.split('T')[0]
        if end_date:
            end_date = end_date.split('T')[0]

        # 1) Data fetch
        raw_df = fetch_data_yahoo(coin_id, start=start_date, end=end_date)
        status_msg = "Data loaded."
        if trigger_id == 'refresh-button':
            status_msg = "Data refreshed from Yahoo Finance."

        # 2) Preprocess
        processed_df = preprocess_data(raw_df)
        if len(processed_df) < 30:
            raise ValueError("Insufficient data (minimum 30 days required).")

        # 3) ADF test => stationarity check
        adf_result = adf_test(processed_df['log_return'])
        adf_text = (f"ADF p-value={adf_result['p_value']:.4f}. "
                    f"Stationary? {adf_result['is_stationary']}\n")
        # optional differencing
        differenced = False
        if not adf_result['is_stationary']:
            processed_df['log_return'] = processed_df['log_return'].diff().dropna()
            differenced = True
            adf_text += " => Non-stationary. Applied 1st difference.\n"

        # 4) Train/Test or Future
        if forecast_mode == 'backtest':
            split_index = len(processed_df) - horizon
            if split_index < horizon:
                raise ValueError(f"Insufficient data for a {horizon}-day backtest. Need at least {2 * horizon} days.")
            train_df = processed_df.iloc[:split_index]
            test_df = processed_df.iloc[split_index:]
            forecast_dates = test_df['date'].values
        else:
            train_df = processed_df
            test_df = pd.DataFrame()
            last_date = pd.to_datetime(train_df['date'].iloc[-1])
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )

        # 5) ARIMA/GARCH parameter selection
        if param_mode == 'auto':
            best_params = auto_tune_arima_garch(train_df['log_return'].dropna())
            p, d, q = best_params['arima']
            garch_p, garch_q = best_params['garch']
            param_status = f"Auto: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"
        else:
            param_status = f"Manual: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"

        # 6) Model fit
        try:
            arima_model, garch_model, scale = fit_arima_garch(
                train_df['log_return'].dropna(),
                arima_order=(p, d, q),
                garch_order=(garch_p, garch_q),
                dist=garch_dist,
                rescale_data=True,
                scale_factor=100
            )
        except Exception as e:
            raise ValueError(f"Model fitting error: {e}")

        # 7) Residual analysis
        final_resid = garch_model.std_resid
        lb_result = ljung_box_test(final_resid)
        lb_text = (f"Ljung-Box Q p-value={lb_result['lb_pvalue']:.4f}. "
                   f"White Noise? {lb_result['is_white_noise']}\n")

        arch_result = arch_test(final_resid, lags=12)
        arch_text = (f"Engle's ARCH p-value={arch_result['arch_pvalue']:.4f}. "
                     f"Heteroskedastic? {arch_result['heteroskedastic']}\n")

        # 8) Forecast
        forecast_out = forecast_arima_garch(arima_model, garch_model, horizon, scale)
        if forecast_mode == 'backtest':
            last_train_price = train_df['price'].iloc[-1]
            reconstructed_price = last_train_price * np.exp(forecast_out['mean_return'].cumsum())

        else: # future
            last_price = train_df['price'].iloc[-1]
            reconstructed_price = last_price * np.exp(forecast_out['mean_return']).cumprod()
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': reconstructed_price
        })
        # 9) Performance
        metrics = {}
        if forecast_mode == 'backtest' and not test_df.empty:
            merged = pd.merge(test_df, forecast_df, on='date', how='inner')
            if not merged.empty:
                mae_ = mean_absolute_error(merged['price'], merged['forecast_price'])
                rmse_ = root_mean_squared_error(merged['price'], merged['forecast_price'])
                mape_ = mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
                metrics = {'mae': mae_, 'rmse': rmse_, 'mape': mape_}
                metric_text = f"MAE={mae_:.2f}, RMSE={rmse_:.2f}, MAPE={mape_:.2f}%"
            else:
                metric_text = "Error: No overlapping dates for backtesting."

        else:
            metric_text = f"Forecast {horizon} days (no backtest)"

        # combine stats
        stats = {
            **compute_descriptive_stats(processed_df),
            'model_aic': arima_model.aic + garch_model.aic,
            'model_bic': arima_model.bic + garch_model.bic,
            **metrics
        }

        # 10) Plots
        price_fig = price_plot(train_df, forecast_df, forecast_mode)
        hist_fig = histogram_plot(processed_df)

        # Create Q-Q plot using statsmodels
        plt.clf()
        qq = sm.qqplot(processed_df['log_return'].dropna(), line='45', fit=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')

        # Create Plotly figure and add the Q-Q plot image
        qq_fig = go.Figure()
        qq_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )

        qq_fig.update_layout(
            width=700, height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white',
            title='Q-Q Plot of Log Returns'
        )
        
        # Create ACF plot using statsmodels
        plt.clf()
        plot_acf(processed_df['log_return'].dropna(), lags=40, alpha=0.05)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')

        # Create Plotly figure and add the ACF plot image
        acf_fig = go.Figure()
        acf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )

        acf_fig.update_layout(
            width=700, height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white',
            title='ACF'
        )

        # Create PACF plot using statsmodels
        plt.clf()
        plot_pacf(processed_df['log_return'].dropna(), lags=40, alpha=0.05, method='yw')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')

        # Create Plotly figure and add the PACF plot image
        pacf_fig = go.Figure()
        pacf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )

        pacf_fig.update_layout(
            width=700, height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white',
            title='PACF'
        )

        # residual plot
        resid_fig = go.Figure()
        resid_fig.add_trace(go.Scatter(
            x=np.arange(len(final_resid)),
            y=final_resid,
            mode='lines',
            name='Standardized Residuals'
        ))
        resid_fig.update_layout(
            title="Standardized Residuals (GARCH)",
            xaxis_title="Index",
            yaxis_title="Residual",
            template="plotly_white"
        )

        # tables
        stats_table = create_table_descriptive(stats)
        forecast_table = create_table_forecast(forecast_df)

        diag_message = (
            f"{adf_text}"
            f"{'Differenced log_return.\n' if differenced else ''}"
            f"{lb_text}"
            f"{arch_text}"
        )

        status_full = f"{status_msg} | {param_status} | {metric_text}"
        return (
            status_full,
            price_fig,
            hist_fig,
            qq_fig,
            acf_fig,
            pacf_fig,
            resid_fig,
            stats_table,
            forecast_table,
            diag_message  # Add diagnostics to output
        )

    except ValueError as ve:
        return (
            f"Validation Error: {str(ve)}",
            go.Figure(), go.Figure(), go.Figure(),
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""
        )
    except Exception as e:
        return (
            f"Unexpected Error: {str(e)}",
            go.Figure(), go.Figure(), go.Figure(),
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""
        )

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)