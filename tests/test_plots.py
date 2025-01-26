"""
test_plots.py - Pytest Suite for Financial Visualization Functions

This test suite verifies the functionality of all visualization functions defined in plots.py.
It includes tests for Plotly figures, matplotlib-based plots, and HTML table generation.

Usage:
    To run the tests, execute:
        pytest test_plots.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath('/Users/moe/Documents/UNI/KF_Uni_Graz/Masterarbeit/project'))  
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import html
from plots import (
    price_plot,
    histogram_plot,
    qq_plot,
    acf_plot,
    pacf_plot,
    create_table_descriptive,
    create_table_forecast,
    residual_plot
)

# Fixtures
@pytest.fixture
def sample_price_data():
    """Generate sample price data with dates"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.random.lognormal(mean=0, sigma=0.1, size=100).cumsum()
    return pd.DataFrame({'date': dates, 'price': prices})

@pytest.fixture
def sample_forecast_data():
    """Generate sample forecast data"""
    dates = pd.date_range(start='2023-04-10', periods=10, freq='D')
    return pd.DataFrame({
        'date': dates,
        'forecast_price': np.linspace(105, 110, 10) + np.random.normal(0, 0.5, 10)
    })

@pytest.fixture
def sample_returns_data():
    """Generate sample log returns data"""
    np.random.seed(42)
    returns = pd.DataFrame({
        'log_return': np.random.normal(loc=0, scale=0.01, size=1000)
    })
    returns['price'] = np.exp(returns['log_return'].cumsum())
    return returns

@pytest.fixture
def sample_stats():
    """Sample statistics dictionary for descriptive table"""
    return {
        'price_mean': 150.25,
        'price_std': 15.67,
        'logret_mean': 0.0012,
        'mae': 0.0123,
        'model_aic': 1234.56,
        'model_bic': 1345.67
    }

# Test Cases
def test_price_plot(sample_price_data, sample_forecast_data):
    """Test price_plot with and without forecast data"""
    # Test basic plot
    fig = price_plot(sample_price_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text == 'Price + Forecast'
    
    # Test with forecast data
    fig = price_plot(sample_price_data, sample_forecast_data)
    assert len(fig.data) == 2
    assert fig.data[1].name == 'Forecast'
    
    # Verify data types
    assert fig.data[0].x.dtype == 'datetime64[ns]'
    assert fig.data[0].y.dtype == 'float64'

def test_histogram_plot(sample_returns_data):
    """Test histogram_plot with normal distribution"""
    fig = histogram_plot(sample_returns_data)
    
    # Check trace types
    trace_types = [trace.type for trace in fig.data]
    assert 'histogram' in trace_types
    assert 'scatter' in trace_types  # For KDE and normal
    
    # Check layout properties
    assert fig.layout.xaxis.title.text == 'Log Returns'
    assert fig.layout.yaxis.title.text == 'Density'
    
    # Check KDE calculation
    kde_trace = [t for t in fig.data if t.name == 'Kernel Density'][0]
    assert len(kde_trace.x) > 100  # Should have smooth curve

def test_qq_plot(sample_returns_data):
    """Test Q-Q plot generation"""
    fig = qq_plot(sample_returns_data)
    
    # Verify image embedding
    assert len(fig.layout.images) == 1
    image = fig.layout.images[0]
    assert image.xref == 'paper'
    assert image.yref == 'paper'
    
    # Check layout properties
    assert fig.layout.title.text == 'Q-Q Plot of Log Returns'
    assert fig.layout.width == 700
    assert fig.layout.height == 500

def test_acf_plot(sample_returns_data):
    """Test ACF plot generation"""
    fig = acf_plot(sample_returns_data, lags=30)
    
    # Verify image properties
    assert len(fig.layout.images) == 1
    assert fig.layout.title.text == 'Autocorrelation Function (ACF)'
    assert fig.layout.width == 700

def test_pacf_plot(sample_returns_data):
    """Test PACF plot generation"""
    fig = pacf_plot(sample_returns_data, lags=20)
    
    # Verify image properties
    assert len(fig.layout.images) == 1
    assert fig.layout.title.text == 'Partial Autocorrelation Function (PACF)'
    assert fig.layout.height == 400

def test_create_table_descriptive(sample_stats):
    """Test descriptive statistics table creation"""
    table = create_table_descriptive(sample_stats)
    
    # Check table structure
    assert isinstance(table, html.Table)
    assert len(table.children) == 2  # Thead and Tbody
    assert len(table.children[1].children) == len(sample_stats)
    
    # Verify content formatting
    first_row = table.children[1].children[0]
    assert 'Price Mean' in first_row.children[0].children
    assert '150.25' in first_row.children[1].children

def test_create_table_forecast(sample_forecast_data):
    """Test forecast table creation"""
    table = create_table_forecast(sample_forecast_data)
    
    # Check table structure
    assert isinstance(table, html.Table)
    assert len(table.children[1].children) == len(sample_forecast_data)
    
    # Verify change calculations
    changes = [tr.children[2].children for tr in table.children[1].children]
    assert any('▲' in c or '▼' in c for c in changes)

def test_residual_plot():
    """Test residual plot generation"""
    residuals = pd.Series(np.random.normal(0, 1, 100), 
                         index=pd.date_range('2023-01-01', periods=100))
    fig = residual_plot(residuals)
    
    # Check traces
    assert len(fig.data) == 1
    assert fig.data[0].name == 'Residuals'
    
    # Check zero line
    assert len(fig.layout.shapes) == 1
    zero_line = fig.layout.shapes[0]
    assert zero_line.y0 == zero_line.y1 == 0
    
    # Check styling
    assert fig.layout.template == 'plotly_white'
    assert fig.layout.yaxis.title.text == 'Residual'

# Edge Cases
def test_empty_data_handling():
    """Test functions with empty data inputs"""
    # Test empty price plot
    empty_df = pd.DataFrame(columns=['date', 'price'])
    fig = price_plot(empty_df)
    assert len(fig.data[0].x) == 0
    
    # Test empty forecast table
    empty_forecast = pd.DataFrame(columns=['date', 'forecast_price'])
    table = create_table_forecast(empty_forecast)
    assert len(table.children[1].children) == 0

def test_missing_column_handling(sample_price_data):
    """Test error handling for missing columns"""
    with pytest.raises(KeyError):
        # Remove required 'price' column
        price_plot(sample_price_data.drop(columns=['price']))

# Parameterized Tests
@pytest.mark.parametrize("dist_type", ['normal', 't', 'skewt'])
def test_histogram_distribution_types(sample_returns_data, dist_type):
    """Test histogram plot with different distribution types"""
    fig = histogram_plot(sample_returns_data, dist_type=dist_type)
    title = f'Return Distribution (Assumed: {dist_type})'
    assert fig.layout.title.text == title

if __name__ == "__main__":
    pytest.main(["-v", "test_plots.py"])
    
    