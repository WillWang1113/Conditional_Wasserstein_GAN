import torch
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

def autocor(data, lags, plot=True):
    cor = sm.tsa.acf(data, nlags = lags)
    if plot:
        sm.graphics.tsa.plot_acf(data, lags = lags)
        plt.show()
    return cor

def cdf(data):
    # return CDF function
    return ECDF(data)

def psd(data):
    pxx, freqs = plt.psd(data)
    return pxx, freqs