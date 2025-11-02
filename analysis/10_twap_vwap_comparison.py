import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from twap import TWAPstrategy
from vwap import VWAPstrategy
from orders import Order, OrderBook
from simulategbm import simulate_gbm
import matplotlib.pyplot as plt
import pandas as pd

price_series = simulate_gbm(initial_price= 200)[1]
parent_qty = 1000
num_slices = 8

results = []
for strat_name, StratClass in [
    ("TWAP", TWAPstrategy),
    ("VWAP", VWAPstrategy)
]:
    
    # CREATE ORDERBOOK
    book = OrderBook(price_series=price_series)
    # INITIALISE STRATEGY
    strat = StratClass(parent_qty=parent_qty, num_slices=num_slices)
    # RUN STRATEGY
    strat.run(orderbook=book, price_series=price_series)
    
    # FILLED ORDERS FILTERING
    filled_orders = [
        o for o in book.filled_orders if o.source.lower() == strat_name.lower() and o.fill_price is not None and o.side == 'buy'
    ]

    # METRICS CALCULATION
    total_qty = sum([o.quantity for o in filled_orders])
    total_cost = sum([o.quantity * o.fill_price for o in filled_orders])
    avg_fill_price = total_cost / total_qty if total_qty > 0 else None
    initial_price = price_series[0]
    
    # AVG SLIPPAGE CALCULATION
    avg_slippage = (
        sum([(o.fill_price - initial_price) for o in filled_orders]) / len(filled_orders)
        if filled_orders else None
    )   
    
    results.append({
        "Strategy": strat_name,
        "Total Quantity Filled": total_qty,
        "Total Cost Incurred": total_cost,
        "Average Fill Price": avg_fill_price,
        "Average Slippage vs Initial Price": avg_slippage
    })

# DISPLAY RESULTS
df = pd.DataFrame(results)
print("\nTWAP vs VWAP Strategy Execution Summary:")
print(df)


# PLOTTING RESULTS
plt.figure(figsize=(7,4))
plt.bar(df['Strategy'], df['Average Slippage vs Initial Price'], color=['palegreen', 'powderblue'])
plt.title('Average Slippage Comparison: TWAP vs VWAP')
plt.xlabel('Strategy')
plt.ylabel('Average Slippage vs Initial Price')
plt.show()
