import pytest
from unittest.mock import patch, MagicMock
from dash.testing.application_runners import import_app
import pandas as pd
import numpy as np

@pytest.fixture
def dash_app():
    # Import and return the Dash app from the 'app' module
    return import_app('app')

def test_layout(dash_duo, dash_app):
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()
    # Check the main title
    assert dash_duo.find_element("h1").text == "ARIMA-GARCH Crypto Forecasting Dashboard"
    # Check that all control elements are present
    for cid in [
        'garch-distribution', 'forecast-mode', 'param-mode', 'date-range',
        'crypto-dropdown', 'arima-p', 'arima-d', 'arima-q', 'garch-p', 'garch-q',
        'forecast-horizon', 'run-button', 'refresh-button', 'status-message'
    ]:
        assert dash_duo.find_element(f"#{cid}")
    # Check that all graphs are present
    for gid in ['price-plot', 'hist-plot', 'qq-plot', 'acf-plot', 'pacf-plot', 'resid-plot']:
        assert dash_duo.find_element(f"#{gid}")
    # Check that all tables/diagnostics are present
    for tid in ['stats-table', 'forecast-table', 'diagnostics-summary']:
        assert dash_duo.find_element(f"#{tid}")

@patch('app.auto_tune_arima_garch')
@patch('app.forecast_arima_garch')
@patch('app.fit_arima_garch')
@patch('app.preprocess_data')
@patch('app.fetch_data_yahoo')
def test_toggle_inputs(mock_fetch, mock_preprocess, mock_fit, mock_forecast, mock_auto, dash_duo, dash_app):
    # Prepare mocks
    df = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=100),
        'price': np.random.rand(100) * 100
    })
    df_proc = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=100),
        'log_return': np.random.randn(100)
    })
    mock_fetch.return_value = df
    mock_preprocess.return_value = df_proc
    mock_fit.return_value = (MagicMock(), MagicMock(), 1.0)
    mock_forecast.return_value = pd.DataFrame({'mean_return': np.random.randn(30)})
    mock_auto.return_value = {'arima': (2, 1, 2), 'garch': (1, 1)}

    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()
    manual = dash_duo.find_element("#param-mode input[value='manual']")
    auto = dash_duo.find_element("#param-mode input[value='auto']")

    # In manual mode, the inputs should be enabled
    for inp in ["#arima-p", "#arima-d", "#arima-q", "#garch-p", "#garch-q"]:
        assert not dash_duo.find_element(inp).get_property('disabled')
    # Switch to Auto-Tune: inputs should be disabled
    auto.click()
    dash_duo.wait_for_text_to_equal("#status-message", "Data loaded.")
    for inp in ["#arima-p", "#arima-d", "#arima-q", "#garch-p", "#garch-q"]:
        assert dash_duo.find_element(inp).get_property('disabled')
    # Switch back to manual: inputs should be enabled again
    manual.click()
    dash_duo.wait_for_text_to_equal("#status-message", "Data loaded.")
    for inp in ["#arima-p", "#arima-d", "#arima-q", "#garch-p", "#garch-q"]:
        assert not dash_duo.find_element(inp).get_property('disabled')

@patch('app.auto_tune_arima_garch')
@patch('app.forecast_arima_garch')
@patch('app.fit_arima_garch')
@patch('app.preprocess_data')
@patch('app.fetch_data_yahoo')
def test_run_and_refresh(mock_fetch, mock_preprocess, mock_fit, mock_forecast, mock_auto, dash_duo, dash_app):
    # Setup for the "Run Analysis" test
    df = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=100),
        'price': np.random.rand(100) * 100
    })
    df_proc = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=100),
        'log_return': np.random.randn(100)
    })
    mock_fetch.return_value = df
    mock_preprocess.return_value = df_proc
    mock_fit.return_value = (MagicMock(aic=100, bic=110), MagicMock(aic=50, bic=60), 1.0)
    mock_forecast.return_value = pd.DataFrame({'mean_return': np.random.randn(30)})
    mock_auto.return_value = {'arima': (2, 1, 2), 'garch': (1, 1)}

    dash_duo.start_server(dash_app)
    dash_duo.wait_for_page()
    # Set parameters in manual mode
    dash_duo.find_element("#param-mode input[value='manual']").click()
    for inp, val in [
        ("#arima-p", "1"), ("#arima-d", "0"), ("#arima-q", "1"),
        ("#garch-p", "1"), ("#garch-q", "1"), ("#forecast-horizon", "30")
    ]:
        element = dash_duo.find_element(inp)
        element.clear()
        element.send_keys(val)
    dash_duo.find_element("#run-button").click()
    dash_duo.wait_for_text_to_contain("#status-message", "Data loaded.")
    status = dash_duo.find_element("#status-message").text
    assert "Manual: ARIMA(1,0,1), GARCH(1,1)" in status
    for gid in ['price-plot', 'hist-plot', 'qq-plot', 'acf-plot', 'pacf-plot', 'resid-plot']:
        fig = dash_duo.find_element(f"#{gid}").get_attribute('figure')
        assert fig and fig != '{}'

    # Test for data refresh
    mock_fetch.return_value = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'price': np.random.rand(100) * 200
    })
    mock_preprocess.return_value = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'log_return': np.random.randn(100)
    })
    dash_duo.find_element("#refresh-button").click()
    dash_duo.wait_for_text_to_contain("#status-message", "Data refreshed from Yahoo Finance.")
    status = dash_duo.find_element("#status-message").text
    assert "Data refreshed from Yahoo Finance." in status
    for gid in ['price-plot', 'hist-plot', 'qq-plot', 'acf-plot', 'pacf-plot', 'resid-plot']:
        fig = dash_duo.find_element(f"#{gid}").get_attribute('figure')
        assert fig and fig != '{}'