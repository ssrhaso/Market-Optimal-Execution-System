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


def match_orders(current_price, order, timestep):
    """
    Simulates matching an order against the current market price.
    
    Parameters:
    - current_price: The current market price of the stock.
    - order: An instance of the Order class.
    - timestep: The current time step in the simulation.
    
    Returns:
    - Updated order with status, fill_price, and fill_time if filled.
    """  
    
    # Order already filled or cancelled
    if order.status != 'open':
        return False  

    # Market Order: Fill immediately at current price
    if order.order_type == 'market':
        order.status = 'filled'
        order.fill_price = current_price
        order.fill_time = timestep
        return True
    
    
    # Limit Order: Check if current price meets limit criteria
    elif order.order_type == 'limit':
        # Buy limit order: fill if current price <= limit price
        if order.side == 'buy' and current_price <= order.price:
            order.status = 'filled'
            order.fill_price = current_price
            order.fill_time = timestep
            return True
        # Sell limit order: fill if current price >= limit price
        elif order.side == 'sell' and current_price >= order.price:
            order.status = 'filled'
            order.fill_price = current_price
            order.fill_time = timestep
            return True
    return False  # Order not filled
            


if __name__ == "__main__":
    
    # Sample Price Series (for testing)
    _ , price_series = simulate_gbm()
    
    # Sample Orders:
    
    order_subject = input("Enter order side to test (buy/sell): ").strip().lower()
    if order_subject not in ['buy', 'sell']:
        print("Error: side must be 'buy' or 'sell'")
        exit()

    order_quantity = int(input("Enter order quantity: "))

    order_type = input("Enter order type to test (market/limit): ").strip().lower()
    if order_type not in ['market', 'limit']:
        print("Error: order type must be 'market' or 'limit'")
        exit()

    if order_type == "limit":
        order_price = float(input("Enter limit price: "))
    else:
        order_price = None

    orders = [
        Order(order_subject, order_quantity, order_type, order_price)
    ]
    
    
    
    
    # Simulate order matching over the price series
    for t, price in enumerate(price_series): 
        print(f"\nTime Step {t}, Current Price: {price}")
        
        for order in orders:
            if order.status == 'open':
                filled = match_orders(price, order, t)
                if filled:
                    print(f"Order Filled: {order}")


    # Summary of all orders
    print("\n--- Order Summary ---")
    total_buy = 0.0
    total_sell = 0.0
    for o in orders:
        if o.status == 'filled' and o.fill_price is not None:
            if o.side == 'buy':
                total_buy += o.fill_price * o.quantity
            else:
                total_sell += o.fill_price * o.quantity

    print("\nTotal Buy Cost: ", round(total_buy, 2))
    print("Total Sell Revenue: ", round(total_sell, 2))

            