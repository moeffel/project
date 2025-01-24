# main.py

"""
main.py

Example script showing how to integrate:
- data_loader.py
- model.py

To run:
  > python main.py
"""

import pandas as pd
import numpy as np
from data_loader import fetch_data_yahoo, preprocess_data, compute_descriptive_stats
from model import fit_best_model, forecast_with_interval, calculate_metrics
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 1. Fetch raw data (Bitcoin example)
    coin_id = "bitcoin"
    try:
        raw_df = fetch_data_yahoo(coin_id, start="2020-01-01")
    except ValueError as ve:
        logger.error(f"Data Fetching Error: {ve}")
        return

    # Verify 'date' column exists
    if 'date' not in raw_df.columns:
        logger.error("Error: 'date' column is missing in the raw data.")
        logger.error(f"Raw Data Columns: {raw_df.columns.tolist()}")
        return
    else:
        logger.info(f"Raw Data Columns: {raw_df.columns.tolist()}")

    # 2. Preprocess data (log returns, date index)
    try:
        processed_df = preprocess_data(raw_df)
    except KeyError as ke:
        logger.error(f"Preprocessing Error: {ke}")
        return
    except ValueError as ve:
        logger.error(f"Preprocessing Error: {ve}")
        return

    # Check the processed DataFrame
    logger.info("Processed DataFrame Head:")
    logger.info(f"\n{processed_df.head()}")

    # Ensure 'log_return' exists
    if 'log_return' not in processed_df.columns:
        logger.error("Error: 'log_return' column is missing after preprocessing.")
        return
    else:
        logger.info("'log_return' column exists.")

    # 3. Compute descriptive stats
    try:
        stats = compute_descriptive_stats(processed_df)
        logger.info("\nDescriptive Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
    except Exception as e:
        logger.error(f"Descriptive Statistics Error: {e}")
        return

    # 4. Fit the model
    try:
        model_dict = fit_best_model(processed_df['log_return'], auto_select=True)
    except ValueError as ve:
        logger.error(f"Model Fitting Error: {ve}")
        return
    except Exception as e:
        logger.error(f"Unexpected Model Fitting Error: {e}")
        return

    # 5. Forecast next 30 periods
    try:
        forecast_df = forecast_with_interval(model_dict, steps=30, alpha=0.05)
        logger.info("\nForecast:")
        logger.info(f"\n{forecast_df}")
    except Exception as e:
        logger.error(f"Forecasting Error: {e}")
        return

    # 6. Assign future dates to forecast_df
    try:
        last_date = processed_df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_df.index = forecast_dates
        logger.info("\nForecast with Dates:")
        logger.info(f"\n{forecast_df.head()}")
    except Exception as e:
        logger.error(f"Date Assignment Error: {e}")
        return

    # 7. Optionally, plot the forecast
    try:
        import plotly.graph_objs as go
        import plotly.offline as pyo

        fig = go.Figure()

        # Plot actual log returns
        fig.add_trace(go.Scatter(
            x=processed_df.index,
            y=processed_df['log_return'],
            mode='lines',
            name='Actual Log Return'
        ))

        # Plot forecast mean returns
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['mean_return'],
            mode='lines',
            name='Forecast Mean Return'
        ))

        # Plot confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['lower_ci'],
            mode='lines',
            name='Lower CI',
            line=dict(dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['upper_ci'],
            mode='lines',
            name='Upper CI',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title='Bitcoin Log Return Forecast',
            xaxis_title='Date',
            yaxis_title='Log Return',
            hovermode='x unified',
            template='plotly_white'
        )

        # Save the plot as an HTML file and open it in the browser
        pyo.plot(fig, filename='forecast.html')
        logger.info("\nForecast plot saved as 'forecast.html'.")
    except ImportError:
        logger.error("Plotly is not installed. Install it using 'pip install plotly' to enable plotting.")
    except Exception as e:
        logger.error(f"Plotting Error: {e}")

    # 8. If you have actual future returns to compare, calculate metrics
    # For demonstration, let's simulate some actual future returns
    # In real scenarios, replace this with actual data
    try:
        simulated_actual_returns = np.random.normal(0, 0.02, 30)
        simulated_actual_returns = pd.Series(simulated_actual_returns, index=forecast_df.index)

        metrics = calculate_metrics(simulated_actual_returns, forecast_df['mean_return'])
        logger.info("\nForecast Metrics (Simulated Actuals):")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    except Exception as e:
        logger.error(f"Metrics Calculation Error: {e}")

if __name__ == "__main__":
    main()