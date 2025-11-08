import gym
from gym import spaces          # Define structure and bounds for action and observation spaces
import numpy as np

from orders import OrderBook
from market_simulator import generate_random_order, generate_order_arrivals
from volatility import get_current_volatility
from simulategbm import simulate_gbm


class OptimalExecutionEnv(gym.Env):

    def __init__(self, parent_order_size=1000, time_horizon=100, n_levels=10):
        super(OptimalExecutionEnv, self).__init__()
        
        # Simulation State
        self.parent_order_size = parent_order_size
        self.time_horizon = time_horizon
        self.n_levels = n_levels
        
        self.observation_space = spaces.Box(low = np.inf, high = np.inf, shape = (self.n_levels * 2 + 3,), dtype = np.float32) 
        self.action_space = spaces.Discrete(3)  # Example: 3 discrete actions (e.g., nothing, buy, sell)
        
    
    
    # Reset Simulator State
    def reset(self):
        """Resets Enviroment for new episode
        Returns:
            state (np.array): initial state observation
        """
        
        # 1. INITIALISE ORDER BOOK
        self.order_book = OrderBook()
        
        # 2. GENERATE PRICE SERIES (GBM)
        _, self.price_series = simulate_gbm(initial_price=100.0, time_horizon=self.time_horizon, delta_t=1, mu=0.05/252, sigma=0.2)
        
        # 3. INITIALISE TRACKING VARIABLES 
        self.current_step = 0                                   # Current time step in episode  
        self.inventory = self.parent_order_size                 # Remaining shares to execute    
        self.cash_spent = 0.0                                   # Total cash spent on executions
        self.shares_executed = 0                                # Total shares executed
        self.execution_prices = []                              # Prices at which shares were executed
        self.agent_fill_history = []                            # History of agent fills

        # 4. COMPUTE INITIAL VOLATILITY
        self.volatility_series = get_current_volatility(self.price_series, window=10)
        self.current_volatility = self.volatility_series.iloc[0] if len(self.volatility_series) > 0 else 0.01
        
        # 5. INITIALISE ORDER BOOK WITH LIQUIDITY
        self._seed_order_book()
        
        # 6. SET ORDER BOOK TIME
        self.order_book.set_time(0)
       
        # 7. RETURN INITIAL STATE
        return self._get_state()


    # Seed Order Book with Initial Liquidity
    def _seed_order_book(self):
        """Populate order book with initial limit orders to provide liquidity"""
        current_price = self.price_series[0]
        
        # ADD BUY ORDERS 
        for _ in range(5):
            order = generate_random_order(current_price=current_price, side='buy')
            if order.order_type == 'limit':
                self.order_book.add_order(order)
            
        # ADD SELL ORDERS
        for _ in range(5):
            order = generate_random_order(current_price=current_price, side='sell')
            if order.order_type == 'limit':
                self.order_book.add_order(order)


    
    
    
    # Apply RL Agent Action to Simulation
    def step(self, action):
        pass

    # Return State Vector (best bid/ask, inventory, time left, volatility, etc.)
    def _get_state(self):
        pass

    # Calculate Reward (e.g., negative execution cost, slippage, etc.)
    def _calculate_reward(self):
        pass