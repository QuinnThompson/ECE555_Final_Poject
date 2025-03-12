from arima_files.arima_parameterization import HyperParameters, model_comprehension, further_forecast
from arima_files.basic_model_metrics import create_time_series_metrics
from arima_files.arima_helper import _FORECAST_DAYS
from collections import namedtuple
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_SHOW_TESTING_CHARTS = False
_FILE_NAME = "performance.csv"
_ITEM_LENGTH_CONSTANT = 3


Range = namedtuple('Range', ['minimum', 'maximum'])

p_range = Range(minimum=0, maximum=4)
d_range = Range(minimum=0, maximum=3)
q_range = Range(minimum=0, maximum=4)


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

def no_print(*args, **kwargs):
    pass

def handle_arima():
    """An if statement that prevents multithreading from running the beginning process again."""
    
    best_performance = float("inf")
    best_hyperparameters: HyperParameters
    
    working_directory = os.getcwd()
    training_data = pd.concat([load_load_data(Path(working_directory + "\\Native_Load_2022.xlsx")), load_load_data(Path(working_directory + "\\Native_Load_2023.xlsx"))])
    validation_data = load_load_data(Path(working_directory + "\\Native_Load_2024.xlsx"))
    
    create_time_series_metrics(training_data)
    p_distance = p_range.maximum-p_range.minimum
    d_distance = d_range.maximum-d_range.minimum
    q_distance = q_range.maximum-q_range.minimum
    
    value_array = np.empty((p_distance, q_distance, d_distance))
    parameter_list = [np.arange(p_range.minimum, p_range.maximum),np.arange(d_range.minimum, d_range.maximum),np.arange(q_range.minimum, q_range.maximum)]
    combinations = itertools.product(*parameter_list)
    for p, d, q in combinations:
        print(
            f"For Autoregressive Terms (p): {p}, " 
            f"Nonseasonal Differences (d): {d}, "
            f"Lagged Forecast Errors (q): {q}"
        )
        # reset magic method for prints
        original_print = __builtins__["print"]
        __builtins__["print"] = no_print
        performance_accumilation = 0
        training_data_copy = training_data.copy()
        validation_data_copy = validation_data.copy()
        # used for length if wanted to check full 
        item_length = min(len(validation_data_copy) - _FORECAST_DAYS, len(training_data_copy) - _FORECAST_DAYS)

        for _ in range(_ITEM_LENGTH_CONSTANT):
            hyperparameters = HyperParameters(p, d, q)
            forecast = further_forecast(training_data_copy, validation_data_copy, hyperparameters, _SHOW_TESTING_CHARTS)
            performance_accumilation += model_comprehension(forecast, validation_data_copy)
            value_to_move = validation_data_copy.head(1)
            
            training_data_copy = training_data_copy.drop(training_data_copy.index[0])
            validation_data_copy = validation_data_copy.drop(validation_data_copy.index[0])
            training_data_copy = pd.concat([training_data_copy, value_to_move])
        
        __builtins__["print"] = original_print    
        performance_mean = performance_accumilation / _ITEM_LENGTH_CONSTANT
        value_array[p, q, d] = performance_mean
        print(performance_mean)
        
        np.savetxt(_FILE_NAME, value_array.reshape(p_distance * q_distance, d_distance), delimiter=',', fmt='%d')
        if performance_mean < best_performance: 
            best_performance = performance_accumilation
            best_hyperparameters = hyperparameters
            
    print(
        "Best HyperParameters - "
        f"(p): {best_hyperparameters.autoregressive_terms}, "
        f"(d): {best_hyperparameters.nonseasonal_differences}, "
        f"(q): {best_hyperparameters.lagged_forecast_errors}"
    )
    hyperparameters = HyperParameters(best_hyperparameters.autoregressive_terms, best_hyperparameters.nonseasonal_differences, best_hyperparameters.lagged_forecast_errors)
    forecast = further_forecast(training_data, validation_data, hyperparameters, True)
    model_comprehension(forecast, validation_data)
    
def run_single(p: int, d: int, q: int):
    hyperparameters = HyperParameters(p, d, q)
    working_directory = os.getcwd()
    training_data = load_load_data(Path(working_directory + "\\Native_Load_2023.xlsx"))
    validation_data = load_load_data(Path(working_directory + "\\Native_Load_2024.xlsx"))
    
    forecast = further_forecast(training_data, validation_data, hyperparameters, True)
    model_comprehension(forecast, validation_data)