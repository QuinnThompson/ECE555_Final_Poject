"""This file tests ARIMA with the ERCOT database."""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from typing import Optional, Callable, Union

from functools import partial
from sklearn.metrics import root_mean_squared_error
from dataclasses import dataclass
from arima_files.arima_helper import _FORECAST_DAYS

_VISIBLE_EXTRA = 1

@dataclass
class HyperParameters():
    """The simplistic parameters for ARIMA.
    """
    autoregressive_terms: int
    nonseasonal_differences: int
    lagged_forecast_errors: int
    
@dataclass
class SHyperParameters():
    """The Seasonal parameters that control how SARIMAX functions.
    """
    autoregressive_terms: int
    nonseasonal_differences: int
    lagged_forecast_errors: int
    series_length: int
    seasonal_autoregressive_terms: int
    seasonal_nonseasonal_differences: int
    seasonal_lagged_forecast_errors: int
    
def further_forecast(
    training_data: pd.Series, 
    validation_data: pd.Series, 
    hyperparameters: Union[HyperParameters, SHyperParameters], 
    show_charts: bool
    ) -> None:
    """A forceful method to ensure that ARIMA is inclusive of previously forecasted info.

    Args:
        training_data: The data to train on for ARIMA.
        validation_data: The data used to gauge how well ARIMA performed.
        hyperparameters: The parameters used to 
        show_charts: _description_
    """
    training_data_copy = training_data.copy()
    validation_data_copy = validation_data.copy()
    new_values = pd.Series()
    for _ in range(_FORECAST_DAYS):
        new_values = pd.concat([new_values, model_list(pd.concat([training_data_copy, new_values]), hyperparameters)])
        validation_data_copy = validation_data_copy.drop(validation_data_copy.index[0])
    
    if show_charts:
        additive_callback = partial(display_forecast, additional_series = new_values)
        display_data(training_data, None)
        display_data(validation_data[:_FORECAST_DAYS+_VISIBLE_EXTRA], additive_callback)
    
    return new_values

def model_list(
    training_data: pd.Series, 
    hyperparameters: Union[HyperParameters, SHyperParameters], 
    validation_data: Optional[pd.Series] = None, 
    exogenous_data: Optional[pd.Series] = None, 
    exogenous_forecast: Optional[pd.Series] = None
) -> None:
    """Run varying models and display their data.

    Args:
        training_data: The data to train the model with.
        validation_data: The data to verify the model works well.
    """
    # (autoreressive terms, nonseasonal differences, lagged forecast errors)
    arima_order=(
        hyperparameters.autoregressive_terms, 
        hyperparameters.nonseasonal_differences, 
        hyperparameters.lagged_forecast_errors
    ) 
    try:
        sarimax_order = (
            hyperparameters.seasonal_autoregressive_terms, 
            hyperparameters.seasonal_nonseasonal_differences, 
            hyperparameters.seasonal_lagged_forecast_errors,
            hyperparameters.series_length
        )
        model = SARIMAX(
            training_data, 
            order=arima_order, 
            seasonal_order=sarimax_order,
            exog = exogenous_data,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    except AttributeError:
        model = ARIMA(training_data, order=arima_order)

    forecast_series = define_model(model, training_data, exogenous_forecast)
    if validation_data is not None:
        additive_callback = partial(display_forecast, additional_series = forecast_series)
        display_data(training_data, None)
        display_data(validation_data[:_FORECAST_DAYS+_VISIBLE_EXTRA], additive_callback)
    return forecast_series

def define_model(model: ARIMA, input_data: pd.Series, exogenous_forecast: Optional[pd.Series] = None) -> pd.Series:
    """Utilize a generic model to generate future values.

    Args:
        model: The ARIMA model to test.
        input_data: The data that is being used in the forecasting.

    Returns:
        pd.Series: The forecasted data.
    """
    results = model.fit()
    forecast = results.forecast(_FORECAST_DAYS, exog=exogenous_forecast)
    print(f"BIC: {results.bic}, AIC: {results.aic}")
    return forecast

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
    performance = root_mean_squared_error(validation_data.values[:len(forecast.values)], forecast.values)
    print(f"RMSE : {performance}")
    return performance
    
