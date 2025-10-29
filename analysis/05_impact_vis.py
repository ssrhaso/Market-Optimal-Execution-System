import numpy as np                  
import matplotlib.pyplot as plt
import seaborn as sns                   # For enhanced visualisations




# Helper functions for Plotting
def plot_slippage_histogram(slippages):
    """
    Plot histogram of slippages vs time for filled orders.
    """
    plt.figure(figsize=(8,5))                                                                                   # Create figure
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
    def __init__(self, side, quantity, order_type, price=None):
        self.side = side                    # 'buy' or 'sell'
        self.quantity = quantity            # number of shares
        self.order_type = order_type        # 'market' or 'limit'
        self.price = price                  # limit price for limit orders
        
        self.status = 'open'               # order status: 'open', 'filled', 'partially filled', 'cancelled'
        self.fill_price = None             # price at which order was filled
        self.fill_time = None              # time at which order was filled

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
    
    
    # Simulate adding and matching orders
    for t, price in enumerate(price_series):
        
        if t % 12 == 0:                         #If timestep is a multiple of 12 (simulate adding a new order every 12 timesteps)
            
            side = np.random.choice(['buy','sell'])                                         # Randomly choose buy or sell (can be adjusted for realism)
            order_type = np.random.choice(['market','limit'], p=[0.3,0.7])                  # 30% market, 70% limit orders (can be adjusted for realism)
            quantity = np.random.randint(50,501)                                              # Random quantity between 50 and 500 shares (can be adjusted for realism)
            lim_price = price + np.random.uniform(-0.5,0.5) if order_type == 'limit' else None  # Limit price near current price (+/- up to $0) (can be adjusted for realism)

            order = Order(side = side, quantity = quantity, order_type = order_type, price = lim_price)                     # Create the order
            book.add_order(order = order)                                                                                   # Add order to the book
        previous_price = price                                                                                              # Update previous price (for spillage calc)


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
            
    # Summary statistics
    filled_count = len(book.filled_orders)                                                      # Total filled orders
    total_orders = filled_count + len(book.bids) + len(book.asks)                               # Total orders placed                  
    avg_slippage = np.nanmean([s for s in slippages if s is not None])                          # Average slippage, ignoring None values
    fill_rate = (100 * filled_count / total_orders) if total_orders else 0                      # % of orders filled    
    market_fill_percent = (100 * market_filled_first / market_total) if market_total else 0     # % of Market orders filled at 'first available' price

    # Summary statistics output
    print(f"\nFilled orders: {filled_count} / {total_orders}")
    print(f"Average slippage: {avg_slippage:.4f}")
    print(f"Fill rate: {fill_rate:.2f}%")
    print(f"% Market orders filled at first price: {market_fill_percent:.2f}%")

    # PLOTTING using seaborn functions defined above
    plot_slippage_histogram(slippages)
    plot_impact_scatter(order_sizes, slippages)
    plt.show()