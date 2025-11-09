import numpy as np

def simulate_gbm(time_horizon=100, delta_t=1, mu=0.0, sigma=0.015, initial_price=200):
    x = np.linspace(0, time_horizon, time_horizon) # Time vector initialised from 0 to time_horizon
    y = np.zeros(time_horizon)                     # Price vector initialised to zeros
    y[0] = initial_price                           # Set initial price
    delta_t = delta_t / 390                        # Convert delta_t to trading days (assuming 390 minutes in a trading day)

    # For each time step (from day 1 to day 251), calculate the price using Geometric Brownian Motion
    for i in range(1, time_horizon):
        
        #Brownian Motion Equation
        y[i] = y[i-1] * np.exp(
            (mu - 0.5 * sigma**2) * delta_t + 
            sigma * np.sqrt(delta_t) * np.random.normal(0, 1)   #np.random.normal(0, 1) - standard normal distribution with mean 0 and std dev 1
        )
    return x, y