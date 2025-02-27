"""This file tests ARIMA with the ERCOT database."""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Callable
from collections import namedtuple
from functools import partial
import numpy as np
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass
import itertools

_FORECAST_DAYS = 7
_VISIBLE_EXTRA = 7

_SHOW_TESTING_CHARTS = False

Range = namedtuple('Range', ['minimum', 'maximum'])

p_range = Range(minimum=0, maximum=5)
d_range = Range(minimum=0, maximum=5)
q_range = Range(minimum=0, maximum=5)


@dataclass
class HyperParameters():
    autoregressive_terms: int
    nonseasonal_differences: int
    lagged_forecast_errors: int

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

def model_list(training_data: pd.Series, validation_data: pd.Series, hyperparameters: HyperParameters, show_charts: bool) -> None:
    """Run varying models and display their data.

    Args:
        training_data: The data to train the model with.
        validation_data: The data to verify the model works well.
    """
    print(
        f"For Autoregressive Terms (p): {hyperparameters.autoregressive_terms}, " 
        f"Nonseasonal Differences (d): {hyperparameters.nonseasonal_differences}, "
        f"Lagged Forecast Errors (q): {hyperparameters.lagged_forecast_errors}"
    )
    # (autoreressive terms, nonseasonal differences, lagged forecast errors)
    forecast_series = define_model(
        ARIMA(training_data, order=(
            hyperparameters.autoregressive_terms, 
            hyperparameters.nonseasonal_differences, 
            hyperparameters.lagged_forecast_errors
            )), 
        training_data
    )
    additive_callback = partial(display_forecast, additional_series = forecast_series)
    if show_charts:
        display_data(training_data, None)
        display_data(validation_data[:len(forecast_series.index)+_VISIBLE_EXTRA], additive_callback)
    return forecast_series

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
    print(f"BIC: {results.bic}, AIC: {results.aic}")
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

def display_forecast(primary_series: pd.Series, additional_series: pd.Series) -> None:
    """Additional tasks to add to the plot.

    Args:
        additional_series: Another series to plot on the graph
    """
    plt.plot(additional_series.index, additional_series.values)
    plt.fill_between(additional_series.index, primary_series.values[:len(additional_series.values)], additional_series.values, color='gray', alpha=0.5, label='Predicted Difference')
    
def model_comprehension(forecast: pd.Series, validation_data: pd.Series) -> float:
    """Rate the performance of the model.

    Args:
        forecast: The forecasted output of the series.
        validation_data: The data to use for validation of the model.

    Returns:
        The performance metric.
    """
    performance = mean_absolute_error(forecast.values, validation_data.values[:len(forecast.values)])
    print(f"MSE : {performance}")
    return performance
    
if __name__ == "__main__":
    """An if statement that prevents multithreading from running the beginning process again."""
    best_performance = float("inf")
    best_hyperparameters: HyperParameters
    
    working_directory = os.getcwd()
    training_data = load_load_data(Path(working_directory + "/Native_Load_2023.xlsx"))
    validation_data = load_load_data(Path(working_directory + "/Native_Load_2024.xlsx"))
    parameter_list = [np.arange(p_range.minimum, p_range.maximum),np.arange(d_range.minimum, d_range.maximum),np.arange(q_range.minimum, q_range.maximum)]
    combinations = itertools.product(*parameter_list)
    for p, d, q in combinations:
        hyperparameters = HyperParameters(p, d, q)
        forecast = model_list(training_data, validation_data, hyperparameters, _SHOW_TESTING_CHARTS)
        performance = model_comprehension(forecast, validation_data)
        if performance < best_performance: 
            best_performance = performance
            best_hyperparameters = hyperparameters
            
    print(
        "Best HyperParameters - "
        f"(p): {best_hyperparameters.autoregressive_terms}, "
        f"(d): {best_hyperparameters.nonseasonal_differences}, "
        f"(q): {best_hyperparameters.lagged_forecast_errors}"
    )
    hyperparameters = HyperParameters(best_hyperparameters.autoregressive_terms, best_hyperparameters.nonseasonal_differences, best_hyperparameters.lagged_forecast_errors)
    forecast = model_list(training_data, validation_data, hyperparameters, True)
    model_comprehension(forecast, validation_data)
    
