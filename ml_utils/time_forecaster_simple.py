"""
Base class with helper methods for different deep learning
time series forecasting models
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from ml_utils.time_series_utils import train_test_split

class TimeSeriesForecasterSimple:
    def __init__(self, cfg=None):
        """
        Simple time series forecaster class

        Parameters:
        -----------
        cfg : tuple
            (shift, offset, avg_type)
            avg_type : "persist", "drift", "mean", "median"
            offset: only with average_forecast (mean or median). Include periodicity with this
                    - eg offset=12 in a monthly dataset gives a monthly climatology
            shift: how far back to go back
                   - with persistence: shift=1 is the usual persistence (use last value for forecast);
                                       shift=2 means take the 2nd-to-last value
        """
        if cfg is None:
            cfg = (1, 1, "persist")
        self.cfg = cfg
        self.shift, self.offset, self.avg_type = cfg
        self.forecast = self.average_forecast
        self.averager = None
        if self.avg_type == "mean":
            self.averager = np.mean
        elif self.avg_type == "median":
            self.averager = np.median
        elif self.avg_type == "persist":
            self.forecast = self.persistence_forecast
        elif self.avg_type == "drift":
            self.forecast = self.drift_forecast

    def persistence_forecast(self, history):
        return history[-self.shift]

    def drift_forecast(self, history):
        slope = (history[-1] - history[0])/(len(history) - 1)
        return history[-1] + slope
        
    # one-step simple forecast
    def average_forecast(self, history):
        # collect values to average
        if self.offset == 1:
            values = history[-self.shift:]
        else:
            # skip bad configs
            if self.shift*self.offset > len(history):
                raise Exception('Config beyond end of data: %d %d' %(
                    self.shift, self.offset))
            # try and collect self.shift values using offset
            values = list()
            for i in range(1, self.shift+1):
                values.append(history[-i*self.offset])
        # check if we can average
        if len(values) < 2:
            raise Exception('Cannot calculate average')
        # mean of last self.shift values
        return self.averager(values)

    def get_predictions(self, data, n_test):
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = list(train)
        # step over each time-step in the test set
        predictions = []
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.forecast(history)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        return predictions

    def plot_predictions(self, data, n_test):
        predictions = self.get_predictions(data, n_test)
        plt.title(str(self.cfg))
        plt.plot(data)
        x = np.arange(len(data))[-len(predictions):]
        plt.plot(x, predictions)
        plt.show()

    # walk-forward validation for univariate data
    def walk_forward_validation(self, data, n_test):
        predictions = self.get_predictions(data, n_test)
        test = data[-n_test:]
        return np.sqrt(mean_squared_error(test, predictions))
