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


if __name__ == "__main__":
    _, price_series = simulate_gbm(initial_price= 252)
    
    # CALCULATE ROLLING VOLATILITY
    window = 10
    volatility = get_current_volatility(price_series, window=window)
    
    # PLOTTING PRICE SERIES AND VOLATILITY
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(price_series, color="powderblue", label='Simulated Price Series')
    ax1.set_ylabel("Price", color="powderblue")
    ax2 = ax1.twinx()
    ax2.plot(volatility, color="salmon", label=f'Rolling Volatility (window={window})')
    ax2.set_ylabel("Volatility", color="salmon")
    plt.title("Simulated Price Series and Rolling Volatility")
    fig.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    
    
    print("\nRolling Volatility (last 10 values):")
    print(volatility.tail(10))
    