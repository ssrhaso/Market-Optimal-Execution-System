import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from simulategbm import simulate_gbm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_current_volatility(price_series, window=10):
    """
    Calculate rolling std.dev of log returns (volatility)
    
    Args:
        price_series (list/array): prices time series
        window (int): rolling window size. Default 20.
    """
    # CONVERT TO PANDAS SERIES
    prices = pd.Series(price_series)
    
    # CALCULATE LOG RETURNS 
    log_returns = np.log(prices).diff()                         # ln(P_t / P_(t+1)) , 
    
    # CALCULATE ROLLING VOLATILITY
    rolling_vol = log_returns.rolling(window=window).std()      # rolling std.dev of log returns 
    return rolling_vol
