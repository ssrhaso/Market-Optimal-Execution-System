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
