from arima_files.arima_parameterization import SHyperParameters, model_comprehension, model_list
from arima_files.basic_model_metrics import create_time_series_metrics
from arima_files.arima_helper import _FORECAST_DAYS
from collections import namedtuple
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np

_SHOW_TESTING_CHARTS = False
_P = 3
_D = 1
_Q = 3
_SEASON_LENGTH = 12
_SP = 1
_SD = 2
_SQ = 0

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
    ncei_data = load_data['TMAX'].resample('M').mean()
    ncei_section = ncei_data['2017-01-01':'2025-01-01']
    return ncei_section
    

def handle_sarimax():
    """An if statement that prevents multithreading from running the beginning process again."""
    
    working_directory = os.getcwd()
    training_data = pd.concat([
        #load_load_data(Path(working_directory + "/Native_Load_2016.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2017.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2018.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2019.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2020.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2021.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2022.xlsx")),
        load_load_data(Path(working_directory + "/Native_Load_2023.xlsx"))
    ])
    exogenious_data = load_exogenious_data(Path(working_directory + "/2013_to_2025_texas_temp.csv"))
    validation_data = load_load_data(Path(working_directory + "/Native_Load_2024.xlsx"))
    
    #create_time_series_metrics(training_data)
    correlation_matrix = np.corrcoef(exogenious_data[:'2024-01-01'], training_data)
    print(correlation_matrix)
    

    hyperparameters = SHyperParameters(_P, _D, _Q, _SEASON_LENGTH, _SP, _SD, _SQ)
    forecast = model_list(training_data, hyperparameters, validation_data, exogenious_data[:'2024-01-01'],  exogenious_data['2024-01-01':].head(_FORECAST_DAYS))
    performance = model_comprehension(forecast, validation_data)
