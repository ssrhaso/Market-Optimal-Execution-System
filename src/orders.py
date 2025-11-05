import heapq                # Fast priority queue operations, maintain best bids/asks in o(log n) time complexity
import time

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
# Implemented using heaps for efficient order matching
class OrderBook:
    def __init__(self, price_series):
        self.bids = []                      # Max Heap (store as -price, timestamp, order) for buy orders
        self.asks = []                      # Min Heap (store as price, timestamp, order) for sell orders
        
        self.filled_orders = []             # List to hold filled orders
        self.price_series = price_series    # Track simulation prices
        self.current_time = 0               # Current time step in simulation
        
    def set_time(self, time_index):
        self.current_time = time_index
        
    def add_order(self,order):
        
        ts = time.time()
        if order.side == 'buy':
            # Invert price for max-heap behavior, if price is None (market order), use -inf to ensure it has least priority for limit matching, include timestamp
            heapq.heappush(self.bids, (-order.price if order.price is not None else float ('-inf'), ts, order))
        
        else:
            # Regular price for min-heap behavior, if price is None (market order), use inf to ensure it has lowest priority for limit matching, include timestamp
            heapq.heappush(self.asks, (order.price if order.price is not None else float ('inf'), ts, order))
        self.match_orders()
        
        
    def match_orders(self):
        # Market Orders or Crossing Limits get filled simultaneously
        while self.bids and self.asks:                                         # Check if both sides have orders
            bid_price, bid_ts, buy = self.bids[0]                              # Highest bid
            ask_price, ask_ts, sell = self.asks[0]                             # Lowest ask
            
            price_bid = -bid_price if bid_price != float('-inf') else None  # Convert back to positive price
            price_ask = ask_price if ask_price != float('inf') else None    # Convert back to positive price

            fill_price = None
            
            #PRICING LOGIC
            
            # MARKET: fill at prevailing market price (current price in price series)
            if (buy.order_type == 'market' or sell.order_type == 'market'):
                fill_price = self.price_series[self.current_time]
            
            # CROSSING LIMITS:
            elif (buy.price is not None and sell.price is not None and buy.price >= sell.price):
                fill_price = price_ask  # Fill at buy limit price
            
            else:
                break  # No match possible
            
            buy.status = sell.status = 'filled'
            buy.fill_price = sell.fill_price = fill_price
            buy.fill_time = sell.fill_time = self.current_time          
                        
            self.filled_orders.extend([buy, sell])
            heapq.heappop(self.bids)
            heapq.heappop(self.asks)


    def get_top_of_book(self):
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask