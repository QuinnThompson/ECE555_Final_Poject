"""This file tests ARIMA with the ERCOT database."""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Callable
from functools import partial

_FORECAST_DAYS = 90

def load_load_data(load_path: Path) -> pd.Series:
    """Load the data and do some pre-processing.

    Args:
        load_path: The path to the load data.
    """
    load_data = pd.read_excel(load_path)
    load_data['Hour Ending'] = load_data['Hour Ending'].str.replace(' 24:00', ' 00:00')
    load_data['Hour Ending'] = load_data['Hour Ending'].str.replace(' DST', '')
    load_data['Hour Ending'] = pd.to_datetime(load_data['Hour Ending'], format='%m/%d/%Y %H:%M')
    load_data.set_index('Hour Ending', inplace=True)

    ercot_daily = load_data['ERCOT'].resample('D').mean()
    return ercot_daily

def model_list(training_data: pd.Series, validation_data: pd.Series) -> None:
    """Run varying models and display their data.

    Args:
        training_data: The data to train the model with.
        validation_data: The data to verify the model works well.
    """
    # # (autoreressive terms, nonseasonal differences, lagged forecast errors)
    forecast_series = define_model(ARIMA(training_data, order=(3, 1, 3)))
    
    additive_callback = partial(display_forecast, additional_series = forecast_series)
    display_data(training_data)
    display_data(validation_data, additive_callback)

def define_model(model: ARIMA, input_data: pd.Series) -> pd.Series:
    """Utilize a generic model to generate future values.

    Args:
        model: The ARIMA model to test.
        input_data: The data that is being used in the forecasting.

    Returns:
        pd.Series: The forecasted data.
    """
    results = model.fit()
    forecast = results.forecast(steps=_FORECAST_DAYS)
    future_dates = [input_data.index[-1] + DateOffset(days=x) for x in range(1,_FORECAST_DAYS+1)]
    return pd.Series(forecast, index=future_dates)


def display_data(series: pd.Series, additive_callback: Optional[Callable]) -> None:
    """Display the data using matplotlib.
    
    Args:
        series: The pandas series to display.
        additive_callback: A callback to provide additional actions for the display.
    """
    plt.figure(figsize=(12,6))
    plt.plot(series.index, series.values)
    plt.title('ERCOT Native Load - Historical Data and 90-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Load (MWh)')

    if additive_callback is not None:
        additive_callback(series)
    plt.legend()
    plt.show()

def display_forecast(primary_series: pd.Series, additional_series: pd.Series):
    """Additional tasks to add to the plot.

    Args:
        additional_series: Another series to plot on the graph
    """
    plt.plot(additional_series.index, additional_series.values)
    

if __name__ == "__main__":
    working_directory = os.getcwd()
    training_data = load_load_data(Path(working_directory + "/Native_Load_2023.xlsx"))
    validation_data = load_load_data(Path(working_directory + "/Native_Load_2024.xlsx"))
    
    model_list(training_data, validation_data)
