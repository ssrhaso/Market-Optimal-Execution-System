import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# MARKET SIMULATION FUNCTIONS AND CLASSES

# Function to simulate Geometric Brownian Motion (GBM) for stock price paths
def simulate_gbm(time_horizon=252, delta_t=1, mu=0.05/252, sigma=0.01, initial_price=200):
    x = np.linspace(0, time_horizon, time_horizon) # Time vector from 0 to 252 days
    y = np.zeros(time_horizon)                     # Price vector initialised to zeros
    y[0] = initial_price                           # Set initial price


    # For each time step (from day 1 to day 251), calculate the price using Geometric Brownian Motion
    for i in range(1, time_horizon):
        
        #Brownian Motion Equation
        y[i] = y[i-1] * np.exp(
            (mu - 0.5 * sigma**2) * delta_t + 
            sigma * np.sqrt(delta_t) * np.random.normal(0, 1)   #np.random.normal(0, 1) - standard normal distribution with mean 0 and std dev 1
        )
    return x, y

# Order class to represent a stock order
class Order:
    #Initialise order with side, quantity, type, and optional price
    def __init__(self, side, quantity, order_type, price=None, source="naive"):
        self.side = side                    # 'buy' or 'sell'
        self.quantity = quantity            # number of shares
        self.order_type = order_type        # 'market' or 'limit'
        self.price = price                  # limit price for limit orders
        self.status = 'open'               # order status: 'open', 'filled', 'partially filled', 'cancelled'
        self.fill_price = None             # price at which order was filled
        self.fill_time = None              # time at which order was filled
        
        self.source = source                # source of the order (for future use)
        
    # String representation of the Order
    def __str__(self):
        if self.order_type == 'market':
            return f"{self.side.capitalize()} {self.quantity} shares @ market price, status: {self.status}"
        else:
            return f"{self.side.capitalize()} {self.quantity} shares @ limit {self.price}, status: {self.status}"

# OrderBook class to manage all orders
class OrderBook:
    def __init__(self, price_series):
        self.bids = []                      # List to hold buy orders
        self.asks = []                      # List to hold sell orders
        self.filled_orders = []             # List to hold filled orders
        self.price_series = price_series    # Track simulation prices
        self.current_time = 0               # Current time step in simulation
        
    def set_time(self, time_index):
        self.current_time = time_index
        
    def add_order(self,order):
        if order.side == 'buy':
            self.bids.append(order)
            self.bids.sort (key=lambda x: (-x.price if x.price else float('-inf')))  # Highest price first
        
        else:
            self.asks.append(order)
            self.asks.sort (key=lambda x: (x.price if x.price else float ('inf')))   # Lowest price first
        self.match_orders()
        
        
    def match_orders(self):
        # Market Orders or Crossing Limits get filled simultaneously
        while self.bids and self.asks:                      # Check if both sides have orders
            buy = self.bids[0]                              # Highest bid
            sell = self.asks[0]                             # Lowest ask
            fill_price = None
            
            #PRICING LOGIC
            
            # MARKET: fill at prevailing market price (current price in price series)
            if (buy.order_type == 'market' or sell.order_type == 'market'):
                fill_price = self.price_series[self.current_time]
            
            # CROSSING LIMITS:
            elif (buy.price is not None and sell.price is not None and buy.price >= sell.price):
                fill_price = buy.price  # Fill at buy limit price
            
            else:
                break  # No match possible
            
            buy.status = sell.status = 'filled'
            buy.fill_price = sell.fill_price = fill_price
            buy.fill_time = sell.fill_time = self.current_time                      
            self.filled_orders.extend([buy, sell])
            self.bids.pop(0)
            self.asks.pop(0)

                    
                    

            



# Functions to handle order placement for each execution method:

# SIMULATE NAIVE STRATEGY
def run_naive_strategy(book, price_series, num_orders, parent_qty):
    """Simulates Naive (REGULAR) Execution Algorithm
     
     Splits large order into smaller chunks
     Places "Market" buy orders at regular intervals until total quantity is reached
    
    Args:
        book (OrderBook object): Simulated tracking of orders
        price_series (list): Series of prices to simulate against
        parent_qty (int): Total quantity to buy
        num_orders (int): Number of orders to place
    """
    
    interval = len(price_series) // num_orders              # Computes in time steps how often to place naive order
    placed = 0                                              # Tracks total quantity placed so far

    # Loops through each time step and corresponding price (two arguments returned by enumerate)
    for t, price in enumerate(price_series):
        side = 'buy'                                        # Specify side 
        order_type = 'market'                               # Specify order type 
        qty = min(50, parent_qty - placed)                  # Determines shares to buy (up to 50 or remaining qty if less than 50) , prevent over-buying    
        order = Order(side = side, quantity = qty, order_type = order_type, price = None, source="naive") # Create order 
        book.set_time(t)                                    # Update current time in order book
        book.add_order(order) # Add order to order book (market)
        placed += qty # Update total placed quantity
        
        # Exit loop if total quantity has been placed
        if placed >= parent_qty:
            break  
        
        
        # SIMULATE SELL ORDERS TO CREATE MARKET ACTIVITY
        sell_qty = np.random.randint(10, 101)               # Random sell quantity between 10 and 100 shares
        sell_order = Order(side = 'sell', quantity = sell_qty, order_type = 'market', price=None, source="naive") # Create sell order
        book.set_time(t)
        book.add_order(sell_order) # Add sell order to order book (market)
        
        
# SIMULATE TWAP STRATEGY
def run_twap_strategy(book, price_series, parent_qty, num_slices):
    """Simulates TWAP (TIME WEIGHTED AVERAGE PRICE) Execution Algorithm
    
    Splits large order into equal sized smaller "slices"
    Submits each "slice" at regular intervals throughout the trading time period

    Args:
        book (OrderBook object): Simulated tracking of orders
        price_series (list): Series of prices to simulate against
        parent_qty (int): Total quantity to buy
        num_slices (int): How many "slices" to split the parent order into (e.g. 8)
    """

    slice_qty = parent_qty // num_slices                                               # Quantity per slice (of parent order)
    twap_intervals = np.linspace(0, len(price_series)-1, num_slices, dtype=int)        # TWAP intervals for order slicing
    qty_remaining = parent_qty                                                         # Track remaining shares to buy

    for i, t in enumerate(twap_intervals):
        
        if i == num_slices - 1:                                                        # For last slice, buy all remaining shares to avoid rounding issues
            qty = qty_remaining
        
        else:
            qty = min(slice_qty, qty_remaining)                                        # Take best of slice quantity or remaining quantity to avoid over-buying
        if qty <= 0:
            break
        
        
        order = Order(side = 'buy', quantity= qty, order_type= 'market', price=None, source="twap") # Create TWAP order
        book.set_time(t)
        book.add_order(order) # Add order to order book (market)
        qty_remaining -= qty                   # Update remaining shares to buy
        
        
        # SIMULATE SELL ORDERS TO CREATE MARKET ACTIVITY
        sell_qty = np.random.randint(10, 101)               # Random sell quantity between 10 and 100 shares
        sell_order = Order(side = 'sell', quantity = sell_qty, order_type = 'market', price=None, source="twap") # Create sell order
        book.set_time(t)
        book.add_order(sell_order) # Add sell order to order book (market)

# SIMULATE VWAP STRATEGY
def run_vwap_strategy(book, price_series, parent_qty, num_slices):
    """Simulates VWAP (VOLUME WEIGHTED AVERAGE PRICE) Execution Algorithm
    
    Splits large order into smaller "slices"
    Submits each "slice" at regular intervals throughout the trading time period
    Each slice quantity is weighted based on simulated market volume at that interval

    Args:
        book (OrderBook object): Simulated tracking of orders
        price_series (list): Series of prices to simulate against
        parent_qty (int): Total quantity to buy
        num_slices (int): How many "slices" to split the parent order into (e.g. 8)
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate simulated "volume profile" (random volumes for each time step)
    
    vol_profile = np.abs(np.random.normal(1, 0.5, len(price_series)))           # Mean 1, std dev 0.5, absolute values to avoid negative volumes assuming 1 is "peak"
    vol_profile /= vol_profile.sum()                                            # Normalise to entirety sum to 1
    shares_remaining = parent_qty                                               # Track remaining shares to buy
    
    for i, v in enumerate(vol_profile):
        qty = int(round(v * parent_qty))            # Proportional quantity based on volume profile
        qty = min(qty, shares_remaining)            # Prevent over-buying
        
        if qty > 0:
            order = Order(side = 'buy', quantity = qty, order_type= 'market', price=None, source="vwap") # Create VWAP order
            book.set_time(i)
            book.add_order(order)                     # Add order to order book (market)
            shares_remaining -= qty                   # Update remaining shares to buy
            
            
            
            # SIMULATE SELL ORDERS TO CREATE MARKET ACTIVITY
            sell_qty = np.random.randint(10, 101)               # Random sell quantity between 10 and 100 shares
            sell_order = Order(side = 'sell', quantity = sell_qty, order_type = 'market', price=None, source="vwap") # Create sell order
            book.set_time(i)
            book.add_order(sell_order)                # Add sell order to order book (market)
            
        if shares_remaining <= 0:
            break
        
# ANALYSIS FOR STRATEGIES
def analyse_strategy(book, price_series, initial_price, parent_qty, source_name):
    filled = [o for o in book.filled_orders if o.source == source_name and o.fill_price is not None and o.side == 'buy']        # Filter filled orders by source name
    total_qty = sum([o.quantity for o in filled])                                                           # Total filled quantity               
    total_cost = sum([o.quantity * o.fill_price for o in filled])                                           # Total cost incurred    
    fill_rate = total_qty / parent_qty if parent_qty > 0 else 0                                             # Fill rate calculation    
    avg_fill_price = (total_cost/ total_qty) if total_qty > 0 else np.nan                                   # Average fill price calculation
    avg_slippage = np.mean([(o.fill_price - initial_price) for o in filled]) if filled else np.nan          # Average slippage calculation
    
    
    return {
        'Strategy' : source_name.upper(),
        'Total Filled Quantity': total_qty,
        'Total Cost': total_cost,
        'Average Fill Price': avg_fill_price,
        'Average Slippage': avg_slippage,   
        'Fill Rate': fill_rate
    }
    
    
    

# MAIN
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    _ , price_series = simulate_gbm(initial_price=200, time_horizon = 252)      # Simulate stock price path
    parent_qty = 1000                                                           # Total quantity to buy (adjustable)
    initial_price = price_series[0]                                             # Initial stock price for slippage calculation
    num_slices = 8                                                              # Number of slices for TWAP and VWAP strategies
    
    
    # RUN & ANALYSE STRATEGIES
    # NAIVE
    book_naive = OrderBook(price_series)
    run_naive_strategy(book_naive, price_series, num_orders=21, parent_qty=parent_qty)
    naive_metrics = analyse_strategy(book_naive, price_series, initial_price, parent_qty, source_name="naive")
    
    # TWAP
    book_twap = OrderBook(price_series)
    run_twap_strategy(book_twap, price_series, parent_qty=parent_qty, num_slices=num_slices)
    twap_metrics = analyse_strategy(book_twap, price_series, initial_price, parent_qty, source_name="twap")
    
    # VWAP
    book_vwap = OrderBook(price_series)
    run_vwap_strategy(book_vwap, price_series, parent_qty=parent_qty, num_slices=num_slices)
    vwap_metrics = analyse_strategy(book_vwap, price_series, initial_price, parent_qty, source_name="vwap")
    
    
    # DISPLAY RESULTS:
    df = pd.DataFrame([naive_metrics, twap_metrics, vwap_metrics])
    df['Total Filled Quantity'] = df['Total Filled Quantity'].astype(int)
    df['Total Cost'] = df['Total Cost'].astype(float)
    df['Average Fill Price'] = df['Average Fill Price'].astype(float)
    df['Average Slippage'] = df['Average Slippage'].astype(float)
    df['Fill Rate'] = df['Fill Rate'].astype(float)

    # Optional: prettier formatting for printing
    df_print = df.copy()
    df_print['Total Cost'] = df_print['Total Cost'].map('{:,.2f}'.format)
    df_print['Average Fill Price'] = df_print['Average Fill Price'].map('{:.2f}'.format)
    df_print['Average Slippage'] = df_print['Average Slippage'].map('{:.2f}'.format)
    df_print['Fill Rate'] = df_print['Fill Rate'].map('{:.3f}'.format)

    print("\nCOST COMPARISON OF EXECUTION STRATEGIES")
    print(df_print.to_string(index=False))

    # Plot 1: Average Fill Price
    plt.figure(figsize=(7, 4))
    plt.bar(df['Strategy'], df['Average Fill Price'], color="#0073c0")
    plt.title('Average Fill Price by Strategy')
    plt.ylabel('Average Fill Price')
    plt.xlabel('Strategy')
    plt.tight_layout()
    plt.show()

    # Plot 2: Average Slippage
    plt.figure(figsize=(7, 4))
    plt.bar(df['Strategy'], df['Average Slippage'], color="#3300ff")
    plt.title('Average Slippage by Strategy')
    plt.ylabel('Average Slippage')
    plt.xlabel('Strategy')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Total Cost
    plt.figure(figsize=(7, 4))
    plt.bar(df['Strategy'], df['Total Cost'], color="#006d33")
    plt.title('Total Cost by Strategy')
    plt.ylabel('Total Cost')
    plt.xlabel('Strategy')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    