import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load and prepare the data
df = pd.read_excel("Native_Load_2024.xlsx")

# Proper datetime conversion with error handling
df['Hour Ending'] = (df['Hour Ending']
                    .str.replace(' 24:00', ' 00:00')
                    .str.replace(' DST', ''))
df['Hour Ending'] = pd.to_datetime(df['Hour Ending'], format='%m/%d/%Y %H:%M')

# Set index and sort
df = df.sort_values('Hour Ending')
df.set_index('Hour Ending', inplace=True)

# Select ERCOT load data
load_series = df['ERCOT'].astype(float)

# ARIMA model with seasonal parameters (p,d,q)x(P,D,Q,s)
# Using order=(2,1,2) and seasonal_order=(1,1,1,24) to capture daily seasonality
model = ARIMA(load_series,
              order=(2,1,2),
              seasonal_order=(1,1,1,24),  # Added seasonal component
              enforce_stationarity=False,
              enforce_invertibility=False)

model_fit = model.fit()

# Forecast for 2025
forecast_horizon = 24 * 365  # One year of hourly forecasts
forecast = model_fit.get_forecast(steps=forecast_horizon)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create future date range
forecast_index = pd.date_range(
    start=load_series.index[-1] + pd.Timedelta(hours=1),
    periods=forecast_horizon,
    freq='H'
)

# Plotting with improved styling
plt.figure(figsize=(16,8))
plt.plot(load_series.index, load_series, 
         label='Historical Load 2024', 
         color='blue', 
         linewidth=1)
plt.plot(forecast_index, forecast_mean, 
         label='2025 Forecast', 
         color='red', 
         linewidth=1)
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1],
                 color='pink',
                 alpha=0.3,
                 label='95% Confidence Interval')

# Improve plot appearance
plt.title("ERCOT Load Forecast 2024-2025", pad=20, size=14)
plt.xlabel("Date", size=12)
plt.ylabel("Load (MW)", size=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.show()
