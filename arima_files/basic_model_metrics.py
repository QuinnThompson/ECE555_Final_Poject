import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
import pandas as pd

_NUMBER_OF_LAGS = 40

def create_time_series_metrics(data_set: pd.Series):
    acf_values = acf(data_set, nlags=_NUMBER_OF_LAGS)
    pacf_values = pacf(data_set, nlags=_NUMBER_OF_LAGS)
    fig, axes = plt.subplots(1, 2)
    axes[0].stem(range(len(acf_values)), acf_values)
    axes[1].stem(range(len(pacf_values)), pacf_values)
    axes[0].set_title('Autocorrelation Function')
    axes[1].set_title('Partial Autocorrelation Function')

    plt.tight_layout()
    plt.show()
    acf_values = acf(data_set.diff().dropna(), nlags=_NUMBER_OF_LAGS)
    pacf_values = pacf(data_set.diff().dropna(), nlags=_NUMBER_OF_LAGS)
    fig, axes = plt.subplots(1, 2)
    axes[0].stem(range(len(acf_values)), acf_values)
    axes[1].stem(range(len(pacf_values)), pacf_values)
    axes[0].set_title('Diff Autocorrelation Function')
    axes[1].set_title('Diff Partial Autocorrelation Function')

    plt.tight_layout()
    plt.show()