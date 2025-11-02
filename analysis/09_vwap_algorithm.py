import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vwap import VWAPstrategy
from orders import Order, OrderBook
from simulategbm import simulate_gbm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    _, price_series = simulate_gbm(initial_price= 252)
    parent_qty = 1000
    num_slices = 8
    
    # CREATE ORDERBOOK
    book = OrderBook(price_series=price_series)
    
    # INITIALISE VWAP STRATEGY
    vwap = VWAPstrategy(parent_qty=parent_qty, num_slices=num_slices)

    # RUN VWAP STRATEGY
    fill_records = vwap.run(orderbook=book, price_series=price_series, parent_qty=parent_qty)


    # OUTPUT RESULTS
    fills_times = [f['time'] for f in fill_records]
    plt.plot(price_series, label='Simulated Price Series')
    plt.scatter(fills_times, [price_series[t] for t in fills_times], c = 'purple', label='VWAP Fill Times')
    plt.title('VWAP Strategy Execution on Simulated Price Series')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    
    # SUMMARY IN TABLE FORMAT
    filled = [o for o in book.filled_orders if o.source == 'vwap' and o.fill_price is not None and o.side == 'buy']     # Filter filled orders by source name
    total_qty = sum([o.quantity for o in filled])                                                                       # Total filled quantity               
    total_cost = sum([o.quantity * o.fill_price for o in filled])                                                       # Total cost incurred    
    avg_fill_price = (total_cost/ total_qty) if total_qty > 0 else np.nan                                               # Average fill price calculation
    initial_price = price_series[0]
    avg_slippage = np.mean([(o.fill_price - initial_price) for o in filled]) if filled else np.nan                      # Average slippage calculation

    print("\nVWAP Strategy Execution Summary:")
    print(f"Total Quantity Filled: {total_qty}")
    print(f"Total Cost Incurred: {total_cost}")
    print(f"Average Fill Price: {avg_fill_price:.2f}")
    print(f"Average Slippage vs Initial Price: {avg_slippage:.2f}")
    
    
    