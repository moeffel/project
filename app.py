"""
app.py

A Dash-based web application for cryptocurrency price analysis and forecasting using ARIMA-GARCH models.

Features:
1. Interactive controls for selecting cryptocurrencies, model parameters, forecast modes, and date ranges.
2. ARIMA-GARCH model fitting with options for manual or auto-tuning of parameters.
3. Forecasting for both backtest and future scenarios.
4. Comprehensive diagnostic tools:
   - ADF Test for stationarity
   - Residual diagnostics (Ljung-Box Q-test, ARCH test)
5. Interactive visualizations:
   - Price trends with forecasts
   - Histogram, Q-Q plot, ACF, PACF, and residual analysis.
6. Performance metrics for backtesting (MAE, RMSE, MAPE).
7. Dynamically updated tables for descriptive statistics and forecasted prices.

Dependencies:
- Dash (with `dcc`, `html`, `Input`, `Output`, `State`)
- Plotly
- Matplotlib
- Statsmodels
- Pandas, NumPy
- Custom modules: `data_loader`, `model`, `utils`, `plots`

Usage:
    Run the script to start the dashboard locally. Use the web interface to select parameters and view results.

Example:
    >>> python crypto_forecasting_dashboard.py
    * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Key Components:
1. **Controls Section:**
   - Select error distribution for GARCH model: Normal, Student's t, or Skewed t.
   - Choose ARIMA and GARCH parameters (manual or auto-tuned).
   - Set forecast mode: backtest or future.
   - Pick cryptocurrency and date range.

2. **Plots:**
   - Price plot with forecasts
   - Histogram of returns
   - Q-Q plot for residual distribution
   - ACF and PACF plots
   - Residual diagnostics

3. **Diagnostics and Tables:**
   - ADF Test results for stationarity.
   - Residual diagnostics (Ljung-Box Q-test, ARCH test).
   - Descriptive statistics and model metrics (AIC, BIC, MAE, RMSE, MAPE).

Functions:
1. `toggle_inputs(mode)`: Enables or disables manual parameter inputs based on selected mode.
2. `update_all_components`: Main callback to handle user interactions, fetch data, fit models, generate diagnostics, and update outputs.

Deployment:
- Use `app.run_server()` for local testing or deploy to platforms like Heroku or AWS.

Example Workflow:
1. Select Bitcoin as the cryptocurrency.
2. Choose a date range (e.g., 2021-01-01 to today).
3. Set ARIMA(1, 1, 1) and GARCH(1, 1) parameters (manual mode).
4. Run the analysis to generate forecasts and diagnostic plots.

Output:
- Interactive visualizations (price, diagnostics).
- Forecasted prices in tabular format.
- Diagnostics summary in plain text.

"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Local modules for data handling, modeling, utilities, and plotting
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

# External stylesheet for Dash app styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # Expose the server variable for deployments

# Mapping of cryptocurrency names to their respective identifiers used in data fetching
CRYPTO_MAP = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Dogecoin (DOGE)': 'dogecoin',
    'Solana (SOL)': 'solana'
}

# Define the layout of the Dash app
app.layout = html.Div([
    # Title of the dashboard
    html.H1("ARIMA-GARCH Crypto Forecasting Dashboard", style={'textAlign': 'center'}),
    
    # ============== CONTROLS SECTION ==============
    html.Div([
        # 1) Error Distribution Selection
        html.Div([
            html.Label("Error Distribution:"),  # Label for the error distribution selection
            dcc.RadioItems(
                id='garch-distribution',  # Unique identifier for the component
                options=[
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': "Student's t", 'value': 't'},
                    {'label': 'Skewed t', 'value': 'skewt'}
                ],
                value='normal',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling for labels
            )
        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container

        # 2) Forecast Mode Selection
        html.Div([
            html.Label("Forecast Mode:"),  # Label for forecast mode selection
            dcc.RadioItems(
                id='forecast-mode',  # Unique identifier
                options=[
                    {'label': ' Backtest', 'value': 'backtest'},
                    {'label': ' Future', 'value': 'future'}
                ],
                value='backtest',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 3) Parameter Mode Selection (Manual or Auto-Tune)
        html.Div([
            html.Label("Parameter Mode:"),  # Label for parameter mode selection
            dcc.RadioItems(
                id='param-mode',  # Unique identifier
                options=[
                    {'label': ' Manual', 'value': 'manual'},
                    {'label': ' Auto-Tune', 'value': 'auto'}
                ],
                value='manual',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 4) Date Range Picker
        html.Div([
            html.Label("Select Date Range:"),  # Label for date range selection
            dcc.DatePickerRange(
                id='date-range',  # Unique identifier
                min_date_allowed=datetime(2015, 1, 1),  # Earliest selectable date
                max_date_allowed=datetime.today(),  # Latest selectable date
                start_date=datetime(2021, 1, 1),  # Default start date
                end_date=datetime.today(),  # Default end date
                display_format='YYYY-MM-DD'  # Format for displaying dates
            )
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container

        # 5) Cryptocurrency Dropdown Selector
        html.Div([
            html.Label("Select Cryptocurrency:"),  # Label for cryptocurrency selection
            dcc.Dropdown(
                id='crypto-dropdown',  # Unique identifier
                options=[{'label': k, 'value': v} for k, v in CRYPTO_MAP.items()],  # Options generated from CRYPTO_MAP
                value='bitcoin',  # Default selection
                clearable=False  # Prevents clearing the selection
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 6) ARIMA Parameters Input (p, d, q)
        html.Div([
            html.Label("ARIMA Parameters:"),  # Label for ARIMA parameters
            html.Div([
                dcc.Input(id='arima-p', type='number', value=1, min=0, 
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for ARIMA p
                dcc.Input(id='arima-d', type='number', value=0, min=0,
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for ARIMA d
                dcc.Input(id='arima-q', type='number', value=1, min=0,
                          style={'width': '60px'})  # Input for ARIMA q
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),  # Styling for the container
        
        # 7) GARCH Parameters Input (p, q)
        html.Div([
            html.Label("GARCH Parameters:"),  # Label for GARCH parameters
            html.Div([
                dcc.Input(id='garch-p', type='number', value=1, min=0,
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for GARCH p
                dcc.Input(id='garch-q', type='number', value=1, min=0,
                          style={'width': '60px'})  # Input for GARCH q
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),  # Styling for the container
        
        # 8) Forecast Horizon Input (Number of Days)
        html.Div([
            html.Label("Forecast Days:"),  # Label for forecast horizon
            dcc.Input(id='forecast-horizon', type='number', value=30, min=1,
                      style={'width': '100px'})  # Input for forecast horizon
        ], style={'width': '10%', 'display': 'inline-block'}),  # Styling for the container
    ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),  # Styling for the controls section
    
    # ============== ACTION SECTION ==============
    html.Div([
        # Button to run the analysis
        html.Button("Run Analysis", id='run-button', n_clicks=0,
                    style={'backgroundColor': '#4CAF50', 'color': 'white', 
                           'padding': '10px 20px', 'borderRadius': '5px'}),  # Styling for the button
        # Button to refresh the data
        html.Button("Refresh Data", id='refresh-button', n_clicks=0,
                    style={'backgroundColor': '#2196F3', 'color': 'white',
                           'padding': '10px 20px', 'marginLeft':'10px',
                           'borderRadius': '5px'}),  # Styling for the button
        # Div to display status messages
        html.Div(id='status-message', style={'color': 'red', 'marginLeft': '20px'})  # Styling for status message
    ], style={'padding': '20px'}),  # Styling for the action section

    # ============== MAIN CONTENT ==============
    html.Div([
        # Left side: Plots
        html.Div([
            # Price Plot
            dcc.Graph(id='price-plot', style={'height': '400px'}),  # Graph component for price plot
            
            # Histogram and Q-Q Plot Side by Side
            html.Div([
                html.Div([dcc.Graph(id='hist-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),  # Histogram plot
                html.Div([dcc.Graph(id='qq-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})  # Q-Q plot
            ]),
            
            # ACF and PACF Plots Side by Side
            html.Div([
                html.Div([dcc.Graph(id='acf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),  # ACF plot
                html.Div([dcc.Graph(id='pacf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})  # PACF plot
            ]),

            # Residual Plot
            dcc.Graph(id='resid-plot', style={'height': '300px'}),  # Residuals plot

        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),  # Styling for the left side
        
        # Right side: Tables & Diagnostics
        html.Div([
            # Performance Metrics Section
            html.H3("Performance Metrics"),  # Header for performance metrics
            html.Div(id='stats-table', style={'marginBottom': '20px'}),  # Container for stats table

            # Forecast Prices Section
            html.H3("Forecast Prices"),  # Header for forecast prices
            html.Div(id='forecast-table'),  # Container for forecast table

            # Diagnostics Section
            html.H3("Diagnostics"),  # Header for diagnostics
            html.Div(id='diagnostics-summary', style={'whiteSpace': 'pre-wrap'})  # Container for diagnostics summary
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 
                  'padding': '20px', 'backgroundColor': '#f9f9f9'})  # Styling for the right side
    ])
])

# ============== CALLBACKS ==============

@app.callback(
    [
        Output('arima-p', 'disabled'),  # Output to disable ARIMA p input
        Output('arima-d', 'disabled'),  # Output to disable ARIMA d input
        Output('arima-q', 'disabled'),  # Output to disable ARIMA q input
        Output('garch-p', 'disabled'),  # Output to disable GARCH p input
        Output('garch-q', 'disabled')   # Output to disable GARCH q input
    ],
    [Input('param-mode', 'value')]  # Input: value of parameter mode radio items
)
def toggle_inputs(mode):
    """
    Toggle the enabled/disabled state of ARIMA and GARCH parameter input fields
    based on the selected parameter mode.

    Parameters:
    - mode (str): The selected parameter mode ('manual' or 'auto').

    Returns:
    - List[bool]: A list indicating whether each input should be disabled.

    Examples:
    >>> toggle_inputs('auto')
    [True, True, True, True, True]

    >>> toggle_inputs('manual')
    [False, False, False, False, False]
    """
    # If mode is 'auto', disable all parameter inputs; otherwise, enable them
    disabled = (mode == 'auto')
    return [disabled, disabled, disabled, disabled, disabled]


@app.callback(
    [
        Output('status-message', 'children'),         # Output for status messages
        Output('price-plot', 'figure'),               # Output for price plot
        Output('hist-plot', 'figure'),                # Output for histogram plot
        Output('qq-plot', 'figure'),                  # Output for Q-Q plot
        Output('acf-plot', 'figure'),                 # Output for ACF plot
        Output('pacf-plot', 'figure'),                # Output for PACF plot
        Output('resid-plot', 'figure'),               # Output for residual plot
        Output('stats-table', 'children'),            # Output for descriptive stats table
        Output('forecast-table', 'children'),         # Output for forecast table
        Output('diagnostics-summary', 'children')     # Output for diagnostics summary
    ],
    [
        Input('run-button', 'n_clicks'),             # Input: clicks on Run Analysis button
        Input('refresh-button', 'n_clicks'),         # Input: clicks on Refresh Data button
        Input('garch-distribution', 'value')          # Input: selected GARCH distribution
    ],
    [
        State('date-range', 'start_date'),            # State: selected start date
        State('date-range', 'end_date'),              # State: selected end date
        State('crypto-dropdown', 'value'),            # State: selected cryptocurrency
        State('param-mode', 'value'),                 # State: parameter mode
        State('forecast-mode', 'value'),              # State: forecast mode
        State('arima-p', 'value'),                     # State: ARIMA p
        State('arima-d', 'value'),                     # State: ARIMA d
        State('arima-q', 'value'),                     # State: ARIMA q
        State('garch-p', 'value'),                     # State: GARCH p
        State('garch-q', 'value'),                     # State: GARCH q
        State('forecast-horizon', 'value')             # State: forecast horizon
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
    Main callback function to handle user interactions and update all components
    of the dashboard accordingly.

    This function performs the following steps:
    1. Determines which button triggered the callback (Run Analysis or Refresh Data).
    2. Fetches cryptocurrency data based on user-selected parameters.
    3. Preprocesses the data and checks for stationarity using the ADF test.
    4. Splits the data into training and testing sets if in backtest mode.
    5. Selects ARIMA and GARCH parameters (manual or auto-tuned).
    6. Fits the ARIMA-GARCH model to the training data.
    7. Performs residual analysis including Ljung-Box and ARCH tests.
    8. Generates forecasts based on the selected mode.
    9. Computes performance metrics if in backtest mode.
    10. Generates all necessary plots and tables for the dashboard.

    Parameters:
    - run_clicks (int): Number of times the Run Analysis button was clicked.
    - refresh_clicks (int): Number of times the Refresh Data button was clicked.
    - garch_dist (str): Selected error distribution for the GARCH model.
    - start_date (str): Start date for data fetching.
    - end_date (str): End date for data fetching.
    - coin_id (str): Identifier of the selected cryptocurrency.
    - param_mode (str): Parameter mode ('manual' or 'auto').
    - forecast_mode (str): Forecast mode ('backtest' or 'future').
    - p (int): ARIMA parameter p.
    - d (int): ARIMA parameter d.
    - q (int): ARIMA parameter q.
    - garch_p (int): GARCH parameter p.
    - garch_q (int): GARCH parameter q.
    - horizon (int): Number of days to forecast.

    Returns:
    - Tuple containing updated status message, figures for all plots, and contents for tables and diagnostics.

    Raises:
    - ValueError: If there are issues with data sufficiency or model fitting.
    - Exception: For any unexpected errors during processing.

    Examples:
    >>> # Since this function relies on Dash's Input and State, it's not directly callable for doctests.
    """
    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        # If no trigger, do not update anything
        return dash.no_update

    # Extract the ID of the triggered input
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        # 1) Data Fetching
        # Convert start and end dates to string format if provided
        if start_date:
            start_date = start_date.split('T')[0]
        if end_date:
            end_date = end_date.split('T')[0]

        # Fetch raw cryptocurrency data from Yahoo Finance or another source
        raw_df = fetch_data_yahoo(coin_id, start=start_date, end=end_date)
        status_msg = "Data loaded."  # Initial status message
        if trigger_id == 'refresh-button':
            status_msg = "Data refreshed from Yahoo Finance."  # Status if refresh button clicked

        # 2) Data Preprocessing
        processed_df = preprocess_data(raw_df)  # Preprocess the raw data
        if len(processed_df) < 30:
            # Ensure there is sufficient data
            raise ValueError("Insufficient data (minimum 30 days required).")

        # 3) Stationarity Check using ADF Test
        adf_result = adf_test(processed_df['log_return'])  # Perform ADF test on log returns
        adf_text = (f"ADF p-value={adf_result['p_value']:.4f}. "
                    f"Stationary? {adf_result['is_stationary']}\n")
        # Apply differencing if data is non-stationary
        differenced = False
        if not adf_result['is_stationary']:
            processed_df['log_return'] = processed_df['log_return'].diff().dropna()  # First difference
            differenced = True
            adf_text += " => Non-stationary. Applied 1st difference.\n"

        # 4) Splitting Data for Backtest or Future Forecast
        if forecast_mode == 'backtest':
            split_index = len(processed_df) - horizon  # Determine split index
            if split_index < horizon:
                # Ensure there's enough data for backtesting
                raise ValueError(f"Insufficient data for a {horizon}-day backtest. Need at least {2 * horizon} days.")
            train_df = processed_df.iloc[:split_index]  # Training data
            test_df = processed_df.iloc[split_index:]   # Testing data
            forecast_dates = test_df['date'].values  # Dates for forecast
        else:
            train_df = processed_df  # Use all data for training in future forecast mode
            test_df = pd.DataFrame()  # No testing data
            last_date = pd.to_datetime(train_df['date'].iloc[-1])  # Last date in training data
            # Generate future dates for forecasting
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )

        # 5) ARIMA/GARCH Parameter Selection
        if param_mode == 'auto':
            # Auto-tune parameters using a predefined function
            best_params = auto_tune_arima_garch(train_df['log_return'].dropna())
            p, d, q = best_params['arima']  # Extract ARIMA parameters
            garch_p, garch_q = best_params['garch']  # Extract GARCH parameters
            param_status = f"Auto: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"  # Status message
        else:
            # Use manually specified parameters
            param_status = f"Manual: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"

        # 6) Model Fitting with ARIMA and GARCH
        try:
            arima_model, garch_model, scale = fit_arima_garch(
                train_df['log_return'].dropna(),  # Training log returns
                arima_order=(p, d, q),            # ARIMA order
                garch_order=(garch_p, garch_q),    # GARCH order
                dist=garch_dist,                   # Selected error distribution
                rescale_data=True,                 # Whether to rescale data
                scale_factor=100                    # Scale factor for rescaling
            )
        except Exception as e:
            # Handle model fitting errors
            raise ValueError(f"Model fitting error: {e}")

        # 7) Residual Analysis
        final_resid = garch_model.std_resid  # Standardized residuals from GARCH model
        resid_index = train_df.index[-len(final_resid):]  # Matching index for residuals

        # Ensure final_resid is a pandas Series with the correct index
        final_resid = pd.Series(final_resid, index=resid_index)

        # Add 'date' as a level in the MultiIndex for residuals
        if 'date' in train_df.columns:
            final_resid.index = pd.MultiIndex.from_product(
                [train_df['date'][-len(final_resid):], ['value']],
                names=['date', 'dummy']
            )
            final_resid.index = final_resid.index.droplevel('dummy')  # Remove the dummy level

        # Perform Ljung-Box Test for White Noise
        lb_result = ljung_box_test(final_resid)
        lb_text = (f"Ljung-Box Q p-value={lb_result['lb_pvalue']:.4f}. "
                   f"White Noise? {lb_result['is_white_noise']}\n")

        # Perform Engle's ARCH Test for Heteroskedasticity
        arch_result = arch_test(final_resid, lags=12)
        arch_text = (f"Engle's ARCH p-value={arch_result['arch_pvalue']:.4f}. "
                     f"Heteroskedastic? {arch_result['heteroskedastic']}\n")
        
        # Create status messages with checkmarks based on test results
        adf_check = "✅" if adf_result['is_stationary'] else "❌"
        lb_check = "✅" if lb_result['is_white_noise'] else "❌"
        arch_check = "✅" if not arch_result['heteroskedastic'] else "❌"

        # Compile diagnostics message
        diag_message = (
            f"ADF p-value={adf_result['p_value']:.4f}. Stationary? {adf_result['is_stationary']} {adf_check}\n"
            f"{'Differenced log_return.\n' if differenced else ''}"
            f"Ljung-Box Q p-value={lb_result['lb_pvalue']:.4f}. White Noise? {lb_result['is_white_noise']} {lb_check}\n"
            f"Engle's ARCH p-value={arch_result['arch_pvalue']:.4f}. Heteroskedastic? {arch_result['heteroskedastic']} {arch_check}\n"
        )

        # 8) Forecast Generation
        forecast_out = forecast_arima_garch(arima_model, garch_model, horizon, scale)  # Generate forecast
        if forecast_mode == 'backtest':
            # In backtest mode, reconstruct prices based on forecasted log returns
            last_train_price = train_df['price'].iloc[-1]  # Last price in training data
            reconstructed_price = last_train_price * np.exp(forecast_out['mean_return'].cumsum())  # Reconstruct prices
        else:
            # In future forecast mode, project future prices based on forecasted mean returns
            last_price = train_df['price'].iloc[-1]  # Last known price
            reconstructed_price = last_price * np.exp(forecast_out['mean_return']).cumprod()  # Project future prices
        
        # Create a DataFrame for forecasted prices
        forecast_df = pd.DataFrame({
            'date': forecast_dates,  # Dates for forecast
            'forecast_price': reconstructed_price  # Forecasted prices
        })

        # 9) Performance Metrics Calculation
        metrics = {}
        if forecast_mode == 'backtest' and not test_df.empty:
            # Merge actual test data with forecasted data based on dates
            merged = pd.merge(test_df, forecast_df, on='date', how='inner')
            if not merged.empty:
                # Calculate performance metrics
                mae_ = mean_absolute_error(merged['price'], merged['forecast_price'])
                rmse_ = root_mean_squared_error(merged['price'], merged['forecast_price'])
                mape_ = mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
                metrics = {'mae': mae_, 'rmse': rmse_, 'mape': mape_}
                metric_text = f"MAE={mae_:.2f}, RMSE={rmse_:.2f}, MAPE={mape_:.2f}%"
            else:
                metric_text = "Error: No overlapping dates for backtesting."
        else:
            # In future forecast mode, indicate the forecast horizon
            metric_text = f"Forecast {horizon} days (no backtest)"

        # Combine descriptive statistics and model information
        stats = {
            **compute_descriptive_stats(processed_df),  # Descriptive statistics
            'model_aic': arima_model.aic + garch_model.aic,  # Combined AIC from ARIMA and GARCH
            'model_bic': arima_model.bic + garch_model.bic,  # Combined BIC from ARIMA and GARCH
            **metrics  # Performance metrics if applicable
        }

        # 10) Plot Generation
        # Generate Price Plot with Forecast
        price_fig = price_plot(train_df, forecast_df, forecast_mode)
        # Generate Histogram Plot of Log Returns
        hist_fig = histogram_plot(processed_df, garch_dist)

        # Generate Q-Q Plot using Statsmodels and convert to Plotly Figure
        plt.clf()  # Clear the current matplotlib figure
        qq = sm.qqplot(processed_df['log_return'].dropna(), line='45', fit=True)  # Create Q-Q plot
        buf = io.BytesIO()  # Create a buffer to hold the image
        plt.savefig(buf, format='png', bbox_inches='tight')  # Save the plot to the buffer
        plt.close()  # Close the matplotlib figure
        buf.seek(0)  # Rewind the buffer
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')  # Encode the image in base64

        # Create Plotly figure and add the Q-Q plot image
        qq_fig = go.Figure()
        qq_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",  # Image source
                x=0, y=1, xref="paper", yref="paper",  # Positioning
                sizex=1, sizey=1, layer="below"  # Size and layering
            )
        )
        qq_fig.update_layout(
            width=700, height=500,  # Size of the figure
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(t=40, b=40),  # Margins
            template='plotly_white',  # Template for styling
            title='Q-Q Plot of Log Returns'  # Title of the plot
        )
        
        # Generate ACF Plot using Statsmodels and convert to Plotly Figure
        plt.clf()  # Clear the current matplotlib figure
        plot_acf(processed_df['log_return'].dropna(), lags=40, alpha=0.05)  # Create ACF plot
        buf = io.BytesIO()  # Create a buffer
        plt.savefig(buf, format='png', bbox_inches='tight')  # Save the plot to the buffer
        plt.close()  # Close the figure
        buf.seek(0)  # Rewind the buffer
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')  # Encode the image

        # Create Plotly figure and add the ACF plot image
        acf_fig = go.Figure()
        acf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",  # Image source
                x=0, y=1, xref="paper", yref="paper",  # Positioning
                sizex=1, sizey=1, layer="below"  # Size and layering
            )
        )
        acf_fig.update_layout(
            width=700, height=400,  # Size of the figure
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(t=40, b=40),  # Margins
            template='plotly_white',  # Template for styling
            title='ACF'  # Title of the plot
        )

        # Generate PACF Plot using Statsmodels and convert to Plotly Figure
        plt.clf()  # Clear the current matplotlib figure
        plot_pacf(processed_df['log_return'].dropna(), lags=40, alpha=0.05, method='yw')  # Create PACF plot
        buf = io.BytesIO()  # Create a buffer
        plt.savefig(buf, format='png', bbox_inches='tight')  # Save the plot to the buffer
        plt.close()  # Close the figure
        buf.seek(0)  # Rewind the buffer
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')  # Encode the image

        # Create Plotly figure and add the PACF plot image
        pacf_fig = go.Figure()
        pacf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",  # Image source
                x=0, y=1, xref="paper", yref="paper",  # Positioning
                sizex=1, sizey=1, layer="below"  # Size and layering
            )
        )
        pacf_fig.update_layout(
            width=700, height=400,  # Size of the figure
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(t=40, b=40),  # Margins
            template='plotly_white',  # Template for styling
            title='PACF'  # Title of the plot
        )

        # Generate Residual Plot with a Red Line at 0
        resid_fig = residual_plot(final_resid)  # Call to residual_plot function

        # Generate Descriptive Statistics Table
        stats_table = create_table_descriptive(stats)  # Call to create descriptive stats table
        # Generate Forecasted Prices Table
        forecast_table = create_table_forecast(forecast_df)  # Call to create forecast table

        # Compile the full status message
        status_full = f"{status_msg} | {param_status} | {metric_text}"
        return (
            status_full,      # Status message
            price_fig,        # Price plot
            hist_fig,         # Histogram plot
            qq_fig,           # Q-Q plot
            acf_fig,          # ACF plot
            pacf_fig,         # PACF plot
            resid_fig,        # Residual plot
            stats_table,      # Descriptive statistics table
            forecast_table,   # Forecasted prices table
            diag_message      # Diagnostics summary
        )

    except ValueError as ve:
        # Handle known validation errors gracefully
        return (
            f"Validation Error: {str(ve)}",  # Status message
            go.Figure(), go.Figure(), go.Figure(),  # Empty figures
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""  # Empty tables and diagnostics
        )
    except Exception as e:
        # Handle any unexpected errors
        return (
            f"Unexpected Error: {str(e)}",    # Status message
            go.Figure(), go.Figure(), go.Figure(),  # Empty figures
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""  # Empty tables and diagnostics
        )

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)
