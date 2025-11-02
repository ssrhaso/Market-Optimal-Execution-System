import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_twap import AdaptiveTWAP      
from twap import TWAPstrategy
from orders import Order, OrderBook
from simulategbm import simulate_gbm
from volatility import get_current_volatility
import matplotlib.pyplot as plt
import numpy as np


# URGENCY INPUT

urgency = input("Enter urgency level (0 to 100, higher means more urgent/faster trading): ")
try:
    urgency = float(urgency)
    assert 0 <= urgency <= 100
except:
    raise ValueError("Invalid urgency level. Please enter a number between 0 and 100.")
urgency_param = urgency / 100  # Normalise to 0-1 range for calculations


      
    




if __name__ == "__main__":
    # Generate market and parameters
    _, price_series = simulate_gbm(initial_price=200)
    parent_qty = 1000
    num_slices = 8

    # Calculate rolling volatility
    vol_series = get_current_volatility(price_series, window=10)

    # RUN ADAPTIVE TWAP
    book_adapt = OrderBook(price_series=price_series)
    atwap = AdaptiveTWAP(
        parent_qty=parent_qty, num_slices=num_slices, vol_series=vol_series,
        vol_threshold=0.010, min_gap=5, max_gap=10, urgency=urgency_param
    )
    print(f"Running Adaptive TWAP with urgency level: {urgency_param:.2f}")
    adapt_fills = atwap.run(orderbook=book_adapt, price_series=price_series)
    adapt_times = [f['time'] for f in adapt_fills]

    # RUN STANDARD TWAP
    book_twap = OrderBook(price_series=price_series)
    twap = TWAPstrategy(parent_qty=parent_qty, num_slices=num_slices)
    twap_fills = twap.run(orderbook=book_twap, price_series=price_series)
    twap_times = [f['time'] for f in twap_fills]

    # PLOT FOR VISUAL COMPARISON
    plt.plot(price_series, color='gray', label='Simulated Price')
    plt.scatter(twap_times, [price_series[t] for t in twap_times], color='blue', label='TWAP Fills', marker='o')
    plt.scatter(adapt_times, [price_series[t] for t in adapt_times], color='red', label='Adaptive TWAP Fills', marker='x')
    plt.title("Adaptive TWAP vs Standard TWAP Execution")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # NUMERIC COMPARISON
    def summary(book, label):
        filled = [o for o in book.filled_orders if o.side == 'buy' and o.fill_price is not None]
        qty = sum(o.quantity for o in filled)
        cost = sum(o.quantity * o.fill_price for o in filled)
        avgp = cost/qty if qty > 0 else np.nan
        initial = price_series[0]
        slip = np.mean([o.fill_price - initial for o in filled]) if filled else np.nan
        print(f"\n{label} Summary")
        print(f"Total Quantity Filled: {qty}")
        print(f"Total Cost: {cost:.2f}")
        print(f"Average Fill Price: {avgp:.2f}")
        print(f"Average Slippage: {slip:.2f}")

    summary(book_twap, "TWAP")
    summary(book_adapt, "Adaptive TWAP")
