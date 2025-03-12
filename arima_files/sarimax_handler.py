from arima_files.arima_parameterization import SHyperParameters, model_comprehension, model_list
from arima_files.basic_model_metrics import create_time_series_metrics
from collections import namedtuple
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np

_SHOW_TESTING_CHARTS = False

Range = namedtuple('Range', ['minimum', 'maximum'])

p_range = Range(minimum=0, maximum=8)
d_range = Range(minimum=0, maximum=3)
q_range = Range(minimum=0, maximum=8)


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

    ercot_daily = load_data['ERCOT'].resample('M').mean()
    return ercot_daily

def load_exogenious_data(load_path: Path) -> pd.Series:
    load_data = pd.read_csv(load_path)
    load_data['DATE'] = pd.to_datetime(load_data['DATE'], format='%Y-%m-%d')
    load_data.set_index('DATE', inplace=True)
    ncei_data = load_data['TAVG'].resample('M').mean()
    ncei_2021_2024 = ncei_data['2021-01-01':'2025-01-01']
    print()
    

def handle_sarimax():
    """An if statement that prevents multithreading from running the beginning process again."""
    
    best_performance = float("inf")
    best_hyperparameters: SHyperParameters
    
    working_directory = os.getcwd()
    training_data = pd.concat([
        load_load_data(Path(working_directory + "/Native_Load_2021.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2022.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2023.xlsx"))
    ])
    exogenious_data = load_exogenious_data(Path(working_directory + "/2020_2025_temp_texas.csv"))
    validation_data = load_load_data(Path(working_directory + "/Native_Load_2024.xlsx"))
    
    create_time_series_metrics(training_data)
    
    parameter_list = [np.arange(p_range.minimum, p_range.maximum),np.arange(d_range.minimum, d_range.maximum),np.arange(q_range.minimum, q_range.maximum)]
    combinations = itertools.product(*parameter_list)
    for p, d, q in combinations:
        hyperparameters = SHyperParameters(p, d, q, 366)
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
    hyperparameters = SHyperParameters(best_hyperparameters.autoregressive_terms, best_hyperparameters.nonseasonal_differences, best_hyperparameters.lagged_forecast_errors, 366)
    forecast = model_list(training_data, validation_data, hyperparameters, True)
    model_comprehension(forecast, validation_data)
    
