# 1. Project Title

Automated ARIMA-GARCH Model for Cryptocurrency Price Prediction with Dash Web App

# 2. Objective

To develop a dynamic, web-based dashboard using Dash/Plotly that allows users to:
• Select cryptocurrencies (BTC, ETH, DOGE, SOL, etc.) for analysis.
• Visualize price forecasts using the ARIMA-GARCH model.
• Review performance metrics like RMSE, MAE, and MAPE.
The model should automatically adjust to the user's selection of cryptocurrencies, parameter settings, and desired forecast period.

# 3. Functional Requirements
## 3.1 User Interface (UI)

• Interactive Web UI (Dash)
o Cryptocurrency Selection: Drop-down to select multiple cryptocurrencies
(e.g., BTC, ETH, DOGE, SOL).
o Date Range Selection: Input for time frame (e.g., 1 year, 2 years, etc.).
o ARIMA-GARCH Parameter Customization: Allow users to enter ARIMA (p, d, q) and GARCH (p, q) values or select "Auto" to auto-tune parameters.
o Error Distribution Selection: Option to choose from Normal, Student’s t, and Skewed Student’s t distributions for GARCH.
o Interactive Plots: View updated plots as users change parameters or select cryptocurrencies.
o Buttons:
§ Run Model: Start the ARIMA-GARCH model.
§ Refresh Data: Download the latest price data.

## 3.2 Data Collection & Preprocessing

• Data Sources: Automatically fetch historical price data using the CoinGecko API or Yahoo Finance.
• Data Cleaning: Remove missing values and fill gaps using interpolation.
• Data Transformation:
o Convert prices to logarithmic returns to stabilize variance.
o Conduct ADF (Augmented Dickey-Fuller) test to check for stationarity.
o If necessary, apply differencing to make the series stationary.
• Data Split: Split data into 80% training, 20% testing for model evaluation.

## 3.3 Descriptive Statistics & EDA

• Descriptive Statistics: Calculate mean, standard deviation, skewness, and kurtosis for log returns.
• Exploratory Data Analysis (EDA)
o Time Series Plots: Visualize raw prices and log returns.
o Histogram & Q-Q Plots: Show the distribution of returns.
o Autocorrelation Plots (ACF, PACF): Identify lags for ARIMA model tuning.

## 3.4 ARIMA-GARCH Model Implementation

• ARIMA Model:
o Automatically select the best (p, d, q) using AIC/BIC criteria.
o Option for users to input custom ARIMA (p, d, q) parameters.
• GARCH Model:
o Fit GARCH(1,1) to ARIMA residuals by default, but allow customization of (p, q).
o Allow users to choose between error distributions (Normal, Student’s t, Skewed Student’s t).
o Store model residuals for analysis and diagnostics.

## 3.5 Model Validation & Diagnostics
• Stationarity Check: Run ADF test and display results.
• Residual Analysis:
o Check if residuals are white noise (random) using Ljung-Box Q-test.
o Check for heteroskedasticity using Engle’s ARCH test.
o Visualize residuals using plots.

## 3.6 Forecasting & Performance Evaluation

• Forecasting:
o Generate price forecasts for the selected period (e.g., 30 days).
o Plot Actual vs. Forecasted Prices.

• Performance Metrics:
o MAE (Mean Absolute Error): Measures the average error size.
o RMSE (Root Mean Squared Error): Emphasizes larger errors more than MAE.
o MAPE (Mean Absolute Percentage Error): Shows the average prediction error as a percentage.

• Performance Table:
o Display a table with the performance of the ARIMA-GARCH model for each
cryptocurrency.

## 3.7 Visualization (Plots)

• Line Chart: Display actual vs. forecasted prices.
• Residual Plot: Plot the residuals for each model.
• Bar Chart: Compare performance metrics (MAE, MAPE, RMSE) across differentcryptocurrencies.
• Interactive Plots: Update charts in real-time as users select cryptocurrencies, date
ranges, or change model parameters.

# 4. Non-Functional Requirements

Category Requirement
Usability Easy-to-use web interface (Dash)
Maintainability Clean, modularized, and well-documented Python code.
Extensibility New coins can be added by adding a ticker.
Performance Fast processing for 2 years of daily data.
Reproducibility Use a random seed for reproducible results.

# 5. Technical Specifications

Component Tool/Library
Data Collection CoinGecko API, requests, Yahoo Finance
Data Handling pandas, numpy
Time Series Analysis statsmodels (ARIMA)
Volatility Modeling arch (for GARCH)
Visualization Dash, Plotly
Deployment Heroku, Local Machine

# 6. Deliverables

• Interactive Web Dashboard: Deploy on Heroku with a public URL.
• Source Code: Clean and well-documented Python code.
• User Manual: Instructions on how to use the dashboard.
• Project Report: Explanation of methods, results, and implementation steps.
