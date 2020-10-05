import pandas as pd
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

from scipy.stats import boxcox

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def rolling_window(a, window_size):
    """
    Parameters:
    -----------
    a : numpy.ndarray
    window_size : int
        size of the rolling window

    Returns:
    --------
    b : numpy.ndarray
        number of columns is window size
        number of rows is number of times window size can fit into array
    """
    # make sure array is flat
    assert(len(a.shape) <= 1 or a.shape[1] ==1)
    n_samples = a.size
    # special cases
    if window_size == 0:
        return np.zeros((n_samples+1,0))
    if window_size == 1:
        return a.reshape((n_samples,1))
    # usual operation
    b = a.flatten()
    shape = (len(b) - window_size + 1, window_size)
    strides = (b.strides[0],) + b.strides
    return np.lib.stride_tricks.as_strided(b, shape=shape, strides=strides)

def split_sequence(sequence, n_in=1, n_out=1):
    """
    Turn sequence into inputs and outputs for a machine learning problem
    NB n_in or n_out can be 0: can be used in a dependent time series problem

    Parameters:
    -----------
    sequence : numpy.ndarray
    n_in : int
        number of time steps to use as inputs
    n_out : int
        number of time steps to use in outputs
        n_out=1 gives a one-step forecast
        n_out>1 gives a multi-step forecast

    Returns:
    --------
    x : numpy.ndarray
        inputs for ML problem in rows
    y : numpy.ndarray
        outputs for ML problem in rows
    index : numpy.ndarray
        for a given row in x,y, the corresponding element of index
        is the index in sequence of the first element of the output.
        This is used for matching inputs and outputs from different time series
        in multivariate and dependent time series problems.
    """
    # can't use last "n_out" values since won't be able to give a full forecast for those steps
    s = sequence if n_out == 0 else sequence[:-n_out]
    x = rolling_window(s, n_in)
    # can't use first "n_in" values since won't be able to get all the required inputs for those steps
    y = rolling_window(sequence[n_in:], n_out)
    return x, y, np.arange(n_in, len(sequence) +1 -n_out)

def split_multiple_sequences(sequences, n_steps_in, n_steps_out=None):
    """
    Turn sequences into inputs and outputs for a machine learning problem

    Parameters:
    -----------
    sequences : list(numpy.ndarray)
    n_steps : list(int)
        number of time steps to use as inputs for each sequence

    Returns:
    --------
    inputs : list(numpy.ndarray)
        inputs for ML problem in rows
    outputs : list(numpy.ndarray)
        outputs for ML problem in rows
    """
    assert(len(sequences)>0)
    assert(len(sequences) == len(n_steps_in))
    if n_steps_out is None:
        n_steps_out = [1 for n in n_steps_in]
    assert(len(sequences) == len(n_steps_out))
    inputs = []
    outputs = []
    n_samples = np.float32("inf")
    n_out_max = np.max(n_steps_out)
    shifts = []
    for seq, n_in, n_out in zip(sequences, n_steps_in, n_steps_out):
        x, y = split_sequence(seq, n_in=n_in, n_out=n_out)
        n_samples = int(np.min([x.shape[0], n_samples]))
        shifts += [n_out_max - n_out]
        inputs += [x]
        outputs += [y]

    def subsample(inputs, shifts, n_samples):
        out = []
        for x, shift in zip(inputs, shifts):
            if shift == 0:
                out += [x[-n_samples:]]
            else:
                out += [x[-n_samples-shift: -shift]]
        return out

    return [subsample(x, shifts, n_samples) for x in [inputs, outputs]]

def split_sequence_lstm(sequence, length, shift=0):
    """
    LSTM's can't handle time series of more than ~200 time steps at once,
    so we split into shorter samples

    They expect their input to be shaped (n_samples, n_time_steps, n_features)

    Parameters:
    -----------
    sequence : numpy.ndarray
    length : int
        number of time steps in each sample

    Returns:
    --------
    split_seq : numpy.ndarray
        sequence reshaped to (n_samples, length, 1)
    """
    n_in = len(sequence)
    n_samples = n_in//length
    n_out = n_samples*length
    assert(shift + n_out <= n_in)
    return sequence[shift:shift + n_out].reshape((n_samples, length, 1))

def test_split(series, ntest=None, frac=.5):
    """
    Split series into train and test series

    Parameters:
    -----------
    series : pandas.Series
    frac : float
        should be in (0,1)

    Returns:
    --------
    train : numpy.ndarray
        training set
    test : numpy.ndarray
        testing set
    """
    x = series.values.astype('float32')
    nx = len(x)
    if ntest is not None:
        assert(ntest<=nx)
        train_size = nx - ntest
    else:
        train_size = int(len(x) * frac)
    train, test = x[0:train_size], x[train_size:]
    print(f"Train with {len(train)}, test on {len(test)}")
    return train, test

def difference(series, interval=1):
    """
    Create a differenced series, usually to give a stationary time series

    Parameters:
    -----------
    series : pandas.Series
    interval : int
        eg interval=1 will remove a trend 
        eg for a monthly series, interval=12 will subtract the same month for the last year
        to allow for seasonality

    Returns:
    --------
    diff : pandas.Series
    """
    return pd.Series(data=series.values[interval:]-series.values[:-interval],
                  index=series.index[:-interval])

def inverse_difference(difference, init, interval=1):
    """
    extend history with differences eg predicted ones

    Parameters:
    -----------
    difference : array-like
        differences to be added
    init : array-like
        initial conditions (history of un-differenced data)

    Returns:
    --------
    predictions : list
        un-differenced predictions for same times as difference
    """
    nhist = len(init)
    assert(nhist >= interval)
    history = list(init) # copy
    for d in difference:
        history += [d + history[-interval]]
    return history[nhist:]

def boxcox_inverse(value, lam):
    """
    Parameters:
    -----------
    value : float or numpy.ndarray
    lam : boxcox exponent

    Returns:
    --------
    out : type(value)
        same type as input
    """
    if lam == 0:
        return np.exp(value)
    return np.exp(np.log(lam * value + 1) / lam)

def split_boxplots(series, freq='A'):
    """
    Follow evolution of distribution with time by splitting into smaller groupds
    and doing box and whisker plots

    Parameters:
    -----------
    series : pandas.Series
    kwargs for pandas.Grouper
        eg freq="M", freq="A", freq="10YS" to split into months, years or decades
    """
    groups = series.groupby(pd.Grouper(freq=freq))
    df = pd.DataFrame()
    for name, group in groups:
        if freq == "10YS" and len(group.values) == 10:
            df[name.year] = group.values
        elif freq == "A" and len(group.values) == 12:
            df[name.year] = group.values
        elif freq == "M":
            df[name.month] = group.values
    df.boxplot()

def summary_plots(series, boxplots=True, autocorr=True, **kwargs):
    """
    Make summary plots for a time series: line plot, PDF, grouped box and whisker plots,
    autocorrelation and partial autocorrelation plots.

    Parameters:
    -----------
    series : pandas.Series
    boxplots : bool
        call split_boxplots to make grouped box and whisker plots
    autocorr : bool
        plot autocorrelation and partial autocorrelation 
    kwargs for split_boxplots
    """
    series.plot()
    plt.show()
    series.plot(kind='kde')
    plt.show()
    if boxplots:
        split_boxplots(series, **kwargs)
        plt.show()
    if autocorr:
        fig = plot_acf(series)
        fig = plot_pacf(series)

def check_stationary(series):
    """
    Check if a time series is stationary, with an augmented Dickey-Fuller test

    Parameters:
    -----------
    series : pandas.Series
    """
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def plot_fit(train, test, predictions):
    """
    Plot fit of a model to data - make line plot, as well as summary plots
    of the residuals

    Parameters:
    -----------
    train : array-like
    test : array-like
    predictions : array-like
    """
    y = list(train) + list(test)
    yhat = list(train) + list(predictions)
    plt.plot(y)
    plt.plot(yhat)
    plt.show()
    
    residuals = [t - p for t,p in zip(test, predictions)]
    residuals = pd.Series(residuals)
    print('Summary statistics of residuals:')
    print(residuals.describe())
    summary_plots(residuals, boxplots=False, autocorr=False)

def evaluate_predictions(test, predictions):
    """
    make and evaluate persistence forecast

    Parameters:
    -----------
    test : array-like
    predictions : array-like

    Returns:
    --------
    rmse : float
    """
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    return rmse

def evaluate_persistence(train, test, bias=0, plot=True):
    """
    make and evaluate persistence forecast

    Parameters:
    -----------
    train : array-like
    test : array-like
    bias : float
    plot : bool
        plot the fit and residuals

    Returns:
    --------
    rmse : float
    predictions : numpy.ndarray
    """
    predictions = [train[-1]] + list(test[:-1])
    rmse = evaluate_predictions(test, predictions)

    if plot:
        plot_fit(train, test, predictions)
    return rmse, predictions

def evaluate_linear_model_simple(train, test, plot=True):
    """
    fit and evaluate linear model to training set
    NB don't redo fit with each new data point from test set

    Parameters:
    -----------
    train : array-like
    test : array-like
    bias : float
    plot : bool
        plot the fit and residuals

    Returns:
    --------
    rmse : float
    predictions : numpy.ndarray
    """
    ntr = len(train)
    nte = len(test)
    xtr = np.array([np.arange(ntr)]).T
    xte = ntr + np.array([np.arange(nte)]).T
    model_fit = LinearRegression().fit(xtr, train)
    predictions = model_fit.predict(xte)
    rmse = evaluate_predictions(test, predictions)

    if plot:
        plot_fit(train, test, predictions)
    return rmse, predictions

def evaluate_linear_model(train, test, plot=True):
    """
    fit and evaluate linear model to training set
    Redo fit with each new data point from test set - use walk-forward validation

    Parameters:
    -----------
    train : array-like
    test : array-like
    bias : float
    plot : bool
        plot the fit and residuals

    Returns:
    --------
    rmse : float
    predictions : numpy.ndarray
    """
    history = list(train)
    predictions = []
    for x in test:
        # predict
        ntr = len(history)
        xtr = np.array([np.arange(ntr)]).T
        model_fit = LinearRegression().fit(xtr, np.array(history))
        history += [x]
        predictions += [model_fit.predict(np.array([[ntr]]))[0]]
    rmse = evaluate_predictions(test, predictions)

    plot_fit(train, test, predictions)
    return rmse, predictions

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(train, test, arima_order, bias=0, plot=True):
    """
    fit and evaluate ARIMA model to training set
    Redo fit with each new data point from test set - use walk-forward validation

    Parameters:
    -----------
    train : array-like
    test : array-like
    arima_order : tuple or list
        (p,d,q) where p is the AR order, d is the number of differences, q is the MA order
    bias : float
    plot : bool
        plot the fit and residuals

    Returns:
    --------
    rmse : float
    predictions : numpy.ndarray
    """
    # prepare training dataset
    history = list(train)
    # make predictions
    predictions = []
    for x in test:
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        predictions += [bias + model_fit.forecast()[0]]
        history += [x]
    # calculate out of sample error
    print(f'ARIMA{arima_order}')
    rmse = evaluate_predictions(test, predictions)
    
    if plot:
        plot_fit(train, test, predictions)
    return rmse, predictions

def evaluate_arima_models(train, test, orders, plot=True):
    """
    Evaluate combinations of p, d and q values for an ARIMA model

    Parameters:
    -----------
    train : array-like
    test : array-like
    orders : iterable
        eg list, generator, itertools.product

    Returns:
    --------
    best_cfg : order with the lowest RMSE
    rmse : float
        RMSE for best_cfg
    predictions : numpy.ndarray
        predictions for best_cfg
    """
    best_score = float("inf"), None
    best_cfg = None
    predictions = None
    for order in orders:
        try:
            rmse, pred = evaluate_arima_model(train, test, order, plot=False)
            if rmse < best_score:
                best_score =rmse
                best_cfg = order
                predictions = pred
                print('\tBest: ARIMA%s RMSE=%.3f' % (order,rmse))
        except:
            # fit doesn't always work
            continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    if plot and predictions is not None:
        plot_fit(train, test, predictions)
    return best_cfg, best_score, predictions
