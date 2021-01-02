"""
Base class with helper methods for different deep learning
time series forecasting models
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


class TimeSeriesForecasterDL(Sequential):

    def preprocess_input(self, x):
        """
        Preprocess inputs to appropriate shape and type for the DL method being used.
        Typically we reshape from [n_samples, n_timesteps]
        into [n_samples, *self.input_shape[1:]]

        Parameters:
        -----------
        x_in : numpy.ndarray
            raw input, shape of [n_samples, n_timesteps]

        Returns:
        --------
        x_out : numpy.ndarray
            processed input, shape of [n_samples, *self.input_shape[1:]]
        """
        return x.reshape(x.shape[0], *self.input_shape[1:])

    def preprocess_output(self, x):
        """
        Preprocess target test outputs to appropriate shape and type for the DL method being used.
        This is a place-holding function which just returns the input.

        Parameters:
        -----------
        x_in : any
            unprocessed output

        Returns:
        --------
        x_out : any
            processed output
        """
        return np.copy(x)

    def fit(self, x, y, **kwargs):
        """
        Parameters:
        -----------
        x : any
            will be processed by self.preprocess_input before being passed into Sequential.fit
        y : numpy.ndarray
            will be processed by self.preprocess_output before being passed into Sequential.fit

        Returns:
        --------
        history : tensorflow.keras.History
        """
        print('Training')
        return super().fit(
                self.preprocess_input(x), self.preprocess_output(y), **kwargs)

    def predict(self, x, **kwargs):
        """
        Parameters:
        -----------
        x : any
            will be processed by self.preprocess_input before being passed into Sequential.fit

        Returns:
        --------
        y : numpy.ndarray
            shape of [n_samples, n_outputs]
        """
        return super().predict(self.preprocess_input(x), **kwargs)

    def plot_model(self, figname):
        """
        visualise network structure

        Parameters:
        -----------
        figname : str
            name of file to save figure to
        """
        print(f'Saving {figname}')
        plot_model(self, to_file=figname, show_shapes=True)
