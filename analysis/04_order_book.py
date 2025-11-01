import numpy as np



# Stock Price Simulator: (from 01_price_simulation.py)
def simulate_gbm(time_horizon=252, delta_t=1, mu=0.05/252, sigma=0.01, initial_price=152):
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




#  OrderBook class to represent all orders
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


if __name__ == "__main__":
    _, price_series = simulate_gbm()
    book = OrderBook()
    np.random.seed(0)

    # Simulate adding new random orders at some timesteps
    for t, price in enumerate(price_series):
        # Add a random order every 12 timesteps (example)
        if t % 12 == 0:
            side = np.random.choice(['buy', 'sell'])
            order_type = np.random.choice(['market', 'limit'])
            quantity = np.random.randint(1, 10)
            # Set a limit price near current price (+/- up to $2)
            lim_price = price + np.random.uniform(-2, 2) if order_type == 'limit' else None
            order = Order(side, quantity, order_type, lim_price)
            book.add_order(order)
        # Optionally match/clear remaining market orders at the current price

    # METRICS & OUTPUTS
    print("\n--- Filled Orders ---")
    for o in book.filled_orders:
        print(o)

    # Calculate slippage (for limit orders)
    slippages = []
    for o in book.filled_orders:
        if o.order_type == 'limit' and o.fill_price is not None:
            slippages.append(o.fill_price - o.price)
    if slippages:
        print("\nAverage Slippage (limit):", round(np.mean(slippages), 4))
    else:
        print("\nNo limit orders were filled.")

    print("\nFill Rate: ", 100 * len(book.filled_orders) / (len(book.filled_orders) + len(book.bids) + len(book.asks)), "%")
    print("\nRemaining buy orders:", [str(o) for o in book.bids])
    print("Remaining sell orders:", [str(o) for o in book.asks])
