import pytest
from unittest.mock import patch, MagicMock
from dash import Dash
from dash.testing.application_runners import import_app
import plotly.graph_objs as go
import pandas as pd
import numpy as np

@pytest.fixture
def dash_app():
    """
    Fixture to initialize the Dash app for testing.
    
    This fixture uses `import_app` from `dash.testing.application_runners` to import the Dash app
    from the `app.py` module. Ensure that the Dash app is named `app` in `app.py`.
    
    Yields:
        Dash: The Dash app instance.
    """
    app = import_app('app')  # Assumes the Dash app is in app.py and named 'app'
    return app

def test_layout(dash_duo, dash_app):
    """
    Test to verify that all essential components are present in the app's layout.
    
    Args:
        dash_duo (DashDuo): Dash testing fixture for interacting with the app.
        dash_app (Dash): The Dash app instance.
    """
    # Start the Dash app using dash_duo
    dash_duo.start_server(dash_app)

    # Allow the app to load
    dash_duo.wait_for_page()

    # Check for the presence of the main title
    assert dash_duo.find_element("h1").text == "ARIMA-GARCH Crypto Forecasting Dashboard"

    # Check for the presence of all controls
    control_ids = [
        'garch-distribution',
        'forecast-mode',
        'param-mode',
        'date-range',
        'crypto-dropdown',
        'arima-p',
        'arima-d',
        'arima-q',
        'garch-p',
        'garch-q',
        'forecast-horizon',
        'run-button',
        'refresh-button',
        'status-message'
    ]

    for cid in control_ids:
        assert dash_duo.find_element(f"#{cid}") is not None, f"Component with id '{cid}' not found."

    # Check for the presence of all graphs
    graph_ids = [
        'price-plot',
        'hist-plot',
        'qq-plot',
        'acf-plot',
        'pacf-plot',
        'resid-plot'
    ]

    for gid in graph_ids:
        assert dash_duo.find_element(f"#{gid}") is not None, f"Graph with id '{gid}' not found."

    # Check for the presence of tables and diagnostics
    table_ids = [
        'stats-table',
        'forecast-table',
        'diagnostics-summary'
    ]

    for tid in table_ids:
        assert dash_duo.find_element(f"#{tid}") is not None, f"Table with id '{tid}' not found."

@patch('app.auto_tune_arima_garch')
@patch('app.forecast_arima_garch')
@patch('app.fit_arima_garch')
@patch('app.preprocess_data')
@patch('app.fetch_data_yahoo')
def test_toggle_inputs(mock_fetch_data_yahoo, mock_preprocess_data, mock_fit_arima_garch,
                      mock_forecast_arima_garch, mock_auto_tune_arima_garch, dash_duo, dash_app):
    """
    Test the 'toggle_inputs' callback to ensure that ARIMA and GARCH parameter inputs are
    enabled or disabled based on the selected parameter mode ('manual' or 'auto').

    Args:
        mock_fetch_data_yahoo (MagicMock): Mock for fetch_data_yahoo function.
        mock_preprocess_data (MagicMock): Mock for preprocess_data function.
        mock_fit_arima_garch (MagicMock): Mock for fit_arima_garch function.
        mock_forecast_arima_garch (MagicMock): Mock for forecast_arima_garch function.
        mock_auto_tune_arima_garch (MagicMock): Mock for auto_tune_arima_garch function.
        dash_duo (DashDuo): Dash testing fixture for interacting with the app.
        dash_app (Dash): The Dash app instance.
    """
    # Setup mock return values to prevent actual data fetching and processing
    mock_fetch_data_yahoo.return_value = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'price': np.random.rand(100) * 100
    })
    mock_preprocess_data.return_value = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'log_return': np.random.randn(100)
    })
    mock_fit_arima_garch.return_value = (MagicMock(), MagicMock(), 1.0)
    mock_forecast_arima_garch.return_value = pd.DataFrame({
        'mean_return': np.random.randn(30)
    })
    mock_auto_tune_arima_garch.return_value = {
        'arima': (2, 1, 2),
        'garch': (1, 1)
    }

    # Start the Dash app
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()

    # Locate the parameter mode radio items
    param_mode_manual = dash_duo.find_element("#param-mode input[value='manual']")
    param_mode_auto = dash_duo.find_element("#param-mode input[value='auto']")

    # Initially, 'manual' is selected, so inputs should be enabled
    assert not dash_duo.find_element("#arima-p").get_property('disabled'), "ARIMA p input should be enabled."
    assert not dash_duo.find_element("#arima-d").get_property('disabled'), "ARIMA d input should be enabled."
    assert not dash_duo.find_element("#arima-q").get_property('disabled'), "ARIMA q input should be enabled."
    assert not dash_duo.find_element("#garch-p").get_property('disabled'), "GARCH p input should be enabled."
    assert not dash_duo.find_element("#garch-q").get_property('disabled'), "GARCH q input should be enabled."

    # Select 'Auto-Tune' mode
    param_mode_auto.click()
    dash_duo.wait_for_text_to_equal("#status-message", "Data loaded.")

    # Inputs should now be disabled
    assert dash_duo.find_element("#arima-p").get_property('disabled'), "ARIMA p input should be disabled."
    assert dash_duo.find_element("#arima-d").get_property('disabled'), "ARIMA d input should be disabled."
    assert dash_duo.find_element("#arima-q").get_property('disabled'), "ARIMA q input should be disabled."
    assert dash_duo.find_element("#garch-p").get_property('disabled'), "GARCH p input should be disabled."
    assert dash_duo.find_element("#garch-q").get_property('disabled'), "GARCH q input should be disabled."

    # Re-select 'Manual' mode
    param_mode_manual.click()
    dash_duo.wait_for_text_to_equal("#status-message", "Data loaded.")

    # Inputs should be enabled again
    assert not dash_duo.find_element("#arima-p").get_property('disabled'), "ARIMA p input should be enabled."
    assert not dash_duo.find_element("#arima-d").get_property('disabled'), "ARIMA d input should be enabled."
    assert not dash_duo.find_element("#arima-q").get_property('disabled'), "ARIMA q input should be enabled."
    assert not dash_duo.find_element("#garch-p").get_property('disabled'), "GARCH p input should be enabled."
    assert not dash_duo.find_element("#garch-q").get_property('disabled'), "GARCH q input should be enabled."

@patch('app.auto_tune_arima_garch')
@patch('app.forecast_arima_garch')
@patch('app.fit_arima_garch')
@patch('app.preprocess_data')
@patch('app.fetch_data_yahoo')
def test_update_all_components_run_analysis(mock_fetch_data_yahoo, mock_preprocess_data,
                                           mock_fit_arima_garch, mock_forecast_arima_garch,
                                           mock_auto_tune_arima_garch, dash_duo, dash_app):
    """
    Test the 'update_all_components' callback by simulating a user clicking the 'Run Analysis' button.
    
    This test ensures that when 'Run Analysis' is clicked with 'manual' parameter mode,
    the app correctly fetches data, processes it, fits the model, and updates all outputs.

    Args:
        mock_fetch_data_yahoo (MagicMock): Mock for fetch_data_yahoo function.
        mock_preprocess_data (MagicMock): Mock for preprocess_data function.
        mock_fit_arima_garch (MagicMock): Mock for fit_arima_garch function.
        mock_forecast_arima_garch (MagicMock): Mock for forecast_arima_garch function.
        mock_auto_tune_arima_garch (MagicMock): Mock for auto_tune_arima_garch function.
        dash_duo (DashDuo): Dash testing fixture for interacting with the app.
        dash_app (Dash): The Dash app instance.
    """
    # Setup mock return values
    mock_fetch_data_yahoo.return_value = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'price': np.random.rand(100) * 100
    })
    mock_preprocess_data.return_value = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'log_return': np.random.randn(100)
    })
    mock_fit_arima_garch.return_value = (MagicMock(aic=100, bic=110), MagicMock(aic=50, bic=60), 1.0)
    mock_forecast_arima_garch.return_value = pd.DataFrame({
        'mean_return': np.random.randn(30)
    })
    mock_auto_tune_arima_garch.return_value = {
        'arima': (2, 1, 2),
        'garch': (1, 1)
    }

    # Start the Dash app
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()

    # Set necessary inputs
    # Select 'manual' parameter mode to enable parameter inputs
    param_mode_manual = dash_duo.find_element("#param-mode input[value='manual']")
    param_mode_manual.click()

    # Set ARIMA parameters
    arima_p = dash_duo.find_element("#arima-p")
    arima_p.clear()
    arima_p.send_keys("1")

    arima_d = dash_duo.find_element("#arima-d")
    arima_d.clear()
    arima_d.send_keys("0")

    arima_q = dash_duo.find_element("#arima-q")
    arima_q.clear()
    arima_q.send_keys("1")

    # Set GARCH parameters
    garch_p = dash_duo.find_element("#garch-p")
    garch_p.clear()
    garch_p.send_keys("1")

    garch_q = dash_duo.find_element("#garch-q")
    garch_q.clear()
    garch_q.send_keys("1")

    # Set Forecast Horizon
    forecast_horizon = dash_duo.find_element("#forecast-horizon")
    forecast_horizon.clear()
    forecast_horizon.send_keys("30")

    # Click the 'Run Analysis' button
    run_button = dash_duo.find_element("#run-button")
    run_button.click()

    # Wait for the status message to update
    dash_duo.wait_for_text_to_contain("#status-message", "Data loaded.")

    # Verify that status message includes 'Manual' parameters and forecast info
    status_message = dash_duo.find_element("#status-message").text
    assert "Manual: ARIMA(1,0,1), GARCH(1,1)" in status_message
    assert "Forecast 30 days (no backtest)" in status_message

    # Verify that plots are updated (non-empty figures)
    graph_ids = [
        'price-plot',
        'hist-plot',
        'qq-plot',
        'acf-plot',
        'pacf-plot',
        'resid-plot'
    ]

    for gid in graph_ids:
        fig = dash_duo.find_element(f"#{gid}").get_attribute('figure')
        assert fig is not None and fig != '{}', f"Graph '{gid}' was not updated properly."

    # Verify that tables are populated
    stats_table = dash_duo.find_element("#stats-table").text
    forecast_table = dash_duo.find_element("#forecast-table").text
    diagnostics_summary = dash_duo.find_element("#diagnostics-summary").text

    assert "mae" in stats_table.lower() or "rmse" in stats_table.lower() or "mape" in stats_table.lower(), "Stats table not populated correctly."
    assert "forecast_price" in forecast_table.lower(), "Forecast table not populated correctly."
    assert "ADF p-value" in diagnostics_summary, "Diagnostics summary not populated correctly."

@patch('app.auto_tune_arima_garch')
@patch('app.forecast_arima_garch')
@patch('app.fit_arima_garch')
@patch('app.preprocess_data')
@patch('app.fetch_data_yahoo')
def test_update_all_components_refresh_data(mock_fetch_data_yahoo, mock_preprocess_data,
                                           mock_fit_arima_garch, mock_forecast_arima_garch,
                                           mock_auto_tune_arima_garch, dash_duo, dash_app):
    """
    Test the 'update_all_components' callback by simulating a user clicking the 'Refresh Data' button.
    
    This test ensures that when 'Refresh Data' is clicked, the app fetches fresh data and updates
    the status message accordingly.

    Args:
        mock_fetch_data_yahoo (MagicMock): Mock for fetch_data_yahoo function.
        mock_preprocess_data (MagicMock): Mock for preprocess_data function.
        mock_fit_arima_garch (MagicMock): Mock for fit_arima_garch function.
        mock_forecast_arima_garch (MagicMock): Mock for forecast_arima_garch function.
        mock_auto_tune_arima_garch (MagicMock): Mock for auto_tune_arima_garch function.
        dash_duo (DashDuo): Dash testing fixture for interacting with the app.
        dash_app (Dash): The Dash app instance.
    """
    # Setup mock return values
    mock_fetch_data_yahoo.return_value = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'price': np.random.rand(100) * 200
    })
    mock_preprocess_data.return_value = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'log_return': np.random.randn(100)
    })
    mock_fit_arima_garch.return_value = (MagicMock(aic=150, bic=160), MagicMock(aic=60, bic=70), 1.0)
    mock_forecast_arima_garch.return_value = pd.DataFrame({
        'mean_return': np.random.randn(30)
    })
    mock_auto_tune_arima_garch.return_value = {
        'arima': (3, 1, 3),
        'garch': (2, 2)
    }

    # Start the Dash app
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()

    # Click the 'Refresh Data' button
    refresh_button = dash_duo.find_element("#refresh-button")
    refresh_button.click()

    # Wait for the status message to update
    dash_duo.wait_for_text_to_contain("#status-message", "Data refreshed from Yahoo Finance.")

    # Verify that status message includes data refreshed info
    status_message = dash_duo.find_element("#status-message").text
    assert "Data refreshed from Yahoo Finance." in status_message

    # Verify that plots are updated (non-empty figures)
    graph_ids = [
        'price-plot',
        'hist-plot',
        'qq-plot',
        'acf-plot',
        'pacf-plot',
        'resid-plot'
    ]

    for gid in graph_ids:
        fig = dash_duo.find_element(f"#{gid}").get_attribute('figure')
        assert fig is not None and fig != '{}', f"Graph '{gid}' was not updated properly after refreshing data."

    # Verify that tables are populated
    stats_table = dash_duo.find_element("#stats-table").text
    forecast_table = dash_duo.find_element("#forecast-table").text
    diagnostics_summary = dash_duo.find_element("#diagnostics-summary").text

    assert "mae" in stats_table.lower() or "rmse" in stats_table.lower() or "mape" in stats_table.lower(), "Stats table not populated correctly after refreshing data."
    assert "forecast_price" in forecast_table.lower(), "Forecast table not populated correctly after refreshing data."
    assert "ADF p-value" in diagnostics_summary, "Diagnostics summary not populated correctly after refreshing data."