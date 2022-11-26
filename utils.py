import torch
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.spatial.distance import correlation
from scipy.stats import wasserstein_distance
from tslearn.metrics import dtw

def autocor(data, lags, plot=True):
    # compute autocorrelation for lag=[0:lags]
    cor = sm.tsa.acf(data, nlags = lags)
    if plot:
        sm.graphics.tsa.plot_acf(data, lags = lags)
        plt.show()
    return cor

def cdf(data):
    # return CDF function
    return ECDF(data)

def psd(data):
    # compute power spectral density
    pxx, freqs = plt.psd(data)
    return pxx, freqs

def cor_dist(data_1, data_2):
    # compute correlation distance between data_1 & data_2
    return correlation(data_1, data_2)

def ws_dist(data_1, data_2):
    # compute wasserstein distance between data_1 & data_2
    return wasserstein_distance(data_1, data_2)

def dtw_dist(data_1, data_2):
    # compute distance between data_1 & data_2 by Dynamic time warping
    return dtw(data_1, data_2)