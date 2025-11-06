import numpy as np
from orders import Order


def generate_order_arrivals(time_steps, avg_orders_per_step):
    """
    Generate order arrival times based on a Poisson process (random arrivals).
    
    Parameters:
    - time_steps: int, total number of time steps in the simulation
    - avg_orders_per_step: float, average number of orders arriving per time step
    
    Returns:
    - List of int: number of orders at each time step
    """
    return np.random.poisson(lam = avg_orders_per_step, size = time_steps)


def generate_random_order(current_price, side=None):
    """Generate single random order

    Args:
        current_price (float): The current market price.
        side (str, optional): The side of the order ('buy' or 'sell'). Defaults to None.
        
    Returns:
        Order object
    """
    
    if side is None:
        side = np.random.choice(['buy', 'sell'])

    # RANDOM SIDE - Taking 70% limit orders, 30% market orders (can be adjusted)
    order_type = np.random.choice(['limit', 'market'], p=[0.7, 0.3])
    
    # RANDOM QUANTITY - between 10 and 200 shares (can be adjusted)
    quantity = np.random.randint(10, 201)
    
    # RANDOM PRICE - for limit orders, within 1% of current price
    if order_type == 'limit':
        price_offset = np.random.uniform(-0.01, 0.01) * current_price
        price = round(current_price + price_offset, 2)
    else:
        price = None # Market orders have no price
        
    return Order(side=side, quantity=quantity, order_type=order_type, price=price, source="market_sim")
