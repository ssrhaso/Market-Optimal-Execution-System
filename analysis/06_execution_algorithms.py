import numpy as np                  
import matplotlib.pyplot as plt
import seaborn as sns                   # For enhanced visualisations




# Helper functions for Plotting
def plot_slippage_histogram(slippages):
    """
    Plot histogram of slippages vs time for filled orders.
    """
    plt.figure(figsize=(8,5))                                                                                    # Create figure
    sns.histplot([s for s in slippages if s is not None], bins = 20, kde=True, color='skyblue')                    # Plot histogram with KDE curve and 20 bar 'bins'
    plt.title('Slippage Distribution')                                                                          # Bins are adjustable to change granularity
    plt.xlabel('Slippage ($)')
    plt.ylabel('Frequency')
    #plt.show()

def plot_impact_scatter(order_sizes, slippages):
    """
    Plot scatter of order sizes vs slippage for filled orders.
    """
    plt.figure(figsize=(8,5))                                                                                 # Create figure                                       
    # Filter pairs so x and y have the same length (drop entries where slippage is None)
    pairs = [(size, slip) for size, slip in zip(order_sizes, slippages) if slip is not None]
    if not pairs:
        print("No non-None slippage values to plot.")
        return
    xs, ys = zip(*pairs)
    
    sns.scatterplot(x=list(xs), y=list(ys))                                 # Scatter plot where x is order sizes and y is slippages       
    plt.title('Order size vs Slippage')
    plt.xlabel('Order Size (shares)')
    plt.ylabel('Slippage ($)')
    #plt.show()
    
# Stock Price Simulator: (from 01_price_simulation.py)
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
    def __init__(self):
        self.bids = []                      # List to hold buy orders
        self.asks = []                      # List to hold sell orders
        self.filled_orders = []             # List to hold filled orders
        
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

            if (buy.order_type == 'market' or sell.order_type =='market') or (buy.price is not None and sell.price is not None and buy.price >= sell.price):
                # if either is market order or bid price is higher or equal to ask price then fill:
                # initialise fill price
                fill_price = None
                
                
                # Determine fill price based on order types:
                
                if buy.order_type == 'market' and sell.order_type == 'limit':
                    fill_price = sell.price
                
                elif buy.order_type == 'limit' and sell.order_type == 'market':
                    fill_price = buy.price
                
                elif buy.order_type == 'market' and sell.order_type == 'market':
                    fill_price = None
                
                else:
                    fill_price = buy.price  # or sell.price, both are same here
                    
                    
                buy.status = sell.status = 'filled'
                buy.fill_price = sell.fill_price = fill_price
                buy.fill_time = sell.fill_time = None                       # Placeholder for future, can be set if needed
                self.filled_orders.extend([buy, sell])
                self.bids.pop(0)
                self.asks.pop(0)
                
            # No more matches possible
            else:
                break  

#  Data collection lists, storing details of each order for analysis
order_sizes = []
order_types = []
execution_prices = []
intended_prices = []
slippages = []
fill_times = []
market_filled_first = 0
market_total = 0

if __name__ == "__main__":
    
    _, price_series = simulate_gbm()            # price_series = array of simulated stock prices 
    book = OrderBook()                          # Initialise an empty order book
    np.random.seed(42)                          # For reproducibility of random numbers
    previous_price = None                       # To track last traded price


    parent_order_qty = 1000                                                             # Parent order quantity (can be adjusted for realism)
    num_slices = 8                                                                      # Number of slices to break parent order into (can be adjusted for realism)
    slice_qty = parent_order_qty // num_slices                                          # Quantity per slice
    twap_intervals = np.linspace(0, len(price_series)-1, num_slices, dtype=int)        # TWAP intervals for order slicing
    twap_orders_submitted = []                                                          # To track submitted TWAP orders                    
    
    
    
    # Simulate adding and matching orders
    for t, price in enumerate(price_series):
        
        #  Naive/random strategy
        if t % 12 == 0:
            side = np.random.choice(['buy','sell'])
            order_type = np.random.choice(['market','limit'], p=[0.3,0.7])
            quantity = np.random.randint(50,501)
            lim_price = price + np.random.uniform(-0.5,0.5) if order_type == 'limit' else None
            order = Order(side=side, quantity=quantity, order_type=order_type, price=lim_price, source="naive")
            book.add_order(order)

        #  TWAP strategy 
        if t in twap_intervals:
            twap_order = Order(side='buy', quantity=slice_qty, order_type='market', price=None, source="twap")
            book.add_order(twap_order)
            twap_orders_submitted.append(t)
            last_trade_price = price_series[0]
            
            
            
            
            
    # Gather stats for analysis & data plots
    for o in book.filled_orders:
        intended_price = o.price if o.order_type == 'limit' else last_trade_price
        slippage = None
        if o.fill_price is not None and intended_price is not None:
            slippage = o.fill_price - intended_price

        order_sizes.append(o.quantity)
        order_types.append(o.order_type)
        execution_prices.append(o.fill_price if o.fill_price is not None else np.nan)
        intended_prices.append(intended_price)
        slippages.append(slippage)
        fill_times.append(o.fill_time)

        # Count market orders filled at last_trade_price
        if o.order_type == 'market':
            market_total += 1
            if o.fill_price == last_trade_price:
                market_filled_first += 1

        # Always update last_trade_price after every fill (simulate ticker tape in real market)
        if o.fill_price is not None:
            last_trade_price = o.fill_price

    if o.fill_price is not None:
        last_trade_price = o.fill_price

    # Naive vs TWAP Metrics


    # Separate filled orders by source tag
    twap_filled = [o for o in book.filled_orders if o.source == 'twap']
    naive_filled = [o for o in book.filled_orders if o.source == 'naive']

    # TWAP metrics
    twap_slippages = [
        o.fill_price - price_series[twap_orders_submitted[i]]
        for i, o in enumerate(twap_filled) if o.fill_price is not None
    ]
    twap_fill_rate = len(twap_filled) / parent_order_qty if parent_order_qty else 0

    # Naive metrics
    naive_slippages = [
        o.fill_price - (o.price if o.order_type == 'limit' else last_trade_price)
        for o in naive_filled if o.fill_price is not None
    ]
    naive_fill_quantity = sum([o.quantity for o in naive_filled])
    naive_fill_rate = naive_fill_quantity / sum([o.quantity for o in naive_filled]) if naive_filled else 0

    print("\n--- TWAP Strategy ---")
    print(f"Filled Orders: {len(twap_filled)} / {parent_order_qty}")
    print(f"Average Slippage: {np.mean(twap_slippages) if twap_slippages else None:.4f}")
    print(f"Fill Rate: {twap_fill_rate*100:.2f}%")

    print("\n--- Naive Strategy ---")
    print(f"Filled Orders: {naive_fill_quantity}")
    print(f"Average Slippage: {np.mean(naive_slippages) if naive_slippages else None:.4f}")
    print(f"Fill Rate: {naive_fill_rate*100:.2f}%")


    # PLOTTING using seaborn functions defined above
    plot_slippage_histogram(slippages)
    plot_impact_scatter(order_sizes, slippages)
    plt.show()