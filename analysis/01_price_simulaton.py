#Simulates stock price movements using Geometric Brownian Motion (GBM) model 

import numpy as np
import matplotlib.pyplot as plt

#Initial Parameters

time_horizon = 252          # 252 trading days in a year
delta_t = 1                 # Time step in days
mu = 0.05 / time_horizon    # Annualised drift (5% return per year)  Average growth or decline rate
sigma = 0.01                # Daily volatility (1% daily volatility) Usually stock prices are between 1% to 4% daily volatility
initial_price = 152         # Initial stock price


#Equation Setup

def simulate_gbm():
    x = np.linspace(0, time_horizon, time_horizon) # Time vector from 0 to 252 days
    y = np.zeros(time_horizon)                     # Price vector initialised to zeros
    y[0] = initial_price                           # Set initial price2


    # For each time step (from day 1 to day 251), calculate the price using Geometric Brownian Motion
    for i in range(1, time_horizon):
        
        #Brownian Motion Equation
        y[i] = y[i-1] * np.exp(
            (mu - 0.5 * sigma**2) * delta_t + 
            sigma * np.sqrt(delta_t) * np.random.normal(0, 1)   #np.random.normal(0, 1) - standard normal distribution with mean 0 and std dev 1
        )
    return x, y


def plot_simulation(x, y):
    plt.plot(x, y)
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price')
    plt.title('Geometric Brownian Motion Stock Price Simulation')
    plt.show()
    

#Run Simulation
def main():
    
    #Number of simulations to run 
    simulations = 10
    plt.figure(figsize=(10, 6))
    for s in range(simulations):
        x, y = simulate_gbm()
        plot_simulation(x, y)

if __name__ == "__main__":
    main()