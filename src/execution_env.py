import gymnasium as gym         # OpenAI Gym for RL environment structure
from gymnasium import spaces          # Define structure and bounds for action and observation spaces
import numpy as np

from orders import Order
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
        
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (8,), dtype = np.float32) 
        self.action_space = spaces.Discrete(3)  # Example: 3 discrete actions (e.g., nothing, buy, sell)
        
    
    
    # Reset Simulator State
    def reset(self, seed=None, options=None):
        """Resets Enviroment for new episode
        Returns:
            state (np.array): initial state observation
        """
        # ERROR HANDLING FOR SEED
        if seed is not None:
            np.random.seed(seed)
            
        # 1. INITIALISE PRICE SERIES AND ORDER BOOK
        _ , self.price_series = simulate_gbm(initial_price=100.0, time_horizon=self.time_horizon, delta_t=1, mu=0.05/252, sigma=0.2)
        self.order_book = OrderBook(price_series=self.price_series)
        
        # 2. INITIALISE TRACKING VARIABLES 
        self.current_step = 0                                   # Current time step in episode  
        self.inventory = self.parent_order_size                 # Remaining shares to execute    
        self.cash_spent = 0.0                                   # Total cash spent on executions
        self.shares_executed = 0                                # Total shares executed
        self.execution_prices = []                              # Prices at which shares were executed
        self.agent_fill_history = []                            # History of agent fills

        # 3. COMPUTE INITIAL VOLATILITY
        self.volatility_series = get_current_volatility(self.price_series, window=10)
        self.current_volatility = self.volatility_series.iloc[0] if len(self.volatility_series) > 0 else 0.01
        
        # 4. INITIALISE ORDER BOOK WITH LIQUIDITY
        self._seed_order_book()
        
        # 5. SET ORDER BOOK TIME
        self.order_book.set_time(0)
       
        # 6. RETURN INITIAL STATE
        observation = self._get_state()
        info = {}
        return observation, info


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


    # Return State Vector (best bid/ask, inventory, time left, volatility, etc.)
    def _get_state(self):
        """Extract and format current state as feature vector for RL agent 
        
        Returns:
            state (np.array): current state observation vector  
        """
        
        # BOUNDS CHECK (CLAMP STEP TO VALID RANGE)
        valid_step = min(self.current_step, len(self.price_series) - 1)
        
        
        # GET MARKET FEATURES
        current_price = self.price_series[valid_step]
        best_bid, best_ask = self.order_book.get_top_of_book()
        
        # CASE HANDLING
        if best_bid is None or best_bid <= 0:
            best_bid = current_price * 0.99 # Default 1% below current price
        if best_ask is None or best_ask <= 0:
            best_ask = current_price * 1.01 # Default 1% above current price
        if best_bid >= best_ask:
            best_ask = best_bid * 1.01      # Ensure ask > bid
        
        # SPREAD
        spread = best_ask - best_bid
        # NORMALISED DETAILS 
        time_progress = self.current_step / self.time_horizon
        inventory_remaining = self.inventory / self.parent_order_size
        
        # VOLATILITY
        if self.current_step <len(self.volatility_series):
            current_vol = self.volatility_series.iloc[self.current_step]
            if np.isnan(current_vol):
                current_vol = 0.01 # Default 1%
        
        else:
            current_vol = 0.01 # Default 1%
            
            
        # EXECUTION PROGRESS (% of shares executed)
        execution_progress = self.shares_executed / self.parent_order_size
        # AVG EXECUTION PRICE
        if self.shares_executed > 0:
            avg_exec_price = self.cash_spent / self.shares_executed
        else:
            avg_exec_price = current_price 
        
        # PREVENT DIVISON BY 0
        if current_price <= 0:
            current_price = 100.0 # Fallback 
        
        # BUILD STATE VECTOR
        state = np.array([
            best_bid / current_price,               # Normalised best bid
            best_ask / current_price,               # Normalised best ask  
            spread / current_price,                 # Normalised spread
            time_progress,                          # Normalised time progress [0,1] (0=start, 1=end)
            inventory_remaining,                    # Normalised inventory remaining
            current_vol,                            # Normalised current volatility
            execution_progress,                     # Normalised execution progress [0,1]
            avg_exec_price / current_price          # Normalised average execution price
        ], dtype=np.float32)
        
        # FINAL SAFETY CHECK
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        
        return state
    
    
    
    
    
    # Apply RL Agent Action to Simulation
    def step(self, action):
        """Execute 1 Timestep of Enviroment
        
        Action Space:
        0. Do nothing (wait)
        1. Execute 10% of remaining inventory as market order
        2. Execute 5% of remaining inventory as market order

        Args:
            action (int): Action chosen by the agent
            
        Returns:
            observation (np.array): next state observation
            reward (float): reward obtained from action
            done (bool): whether episode has ended
            info (dict): additional information
        """
        
        #1. PARSE ACTION, SUBMIT AGENTS ORDER
        if action == 1 and self.inventory > 0:
            # Execute 10% of remaining inventory
            order_size = max(1, int(self.inventory * 0.10))
            order_size = min(order_size, self.inventory)            # Ensure we don't exceed inventory   
            
            agent_order = Order(side='buy', order_type='market', quantity=order_size, source='rl_agent')     
            self.order_book.add_order(agent_order)
        
        elif action == 2 and self.inventory > 0:
            # Execute 5% of remaining inventory
            order_size = max(1, int(self.inventory * 0.05))
            order_size = min(order_size, self.inventory)            # Ensure we don't exceed inventory
            
            agent_order = Order(side='buy', order_type='market', quantity=order_size, source='rl_agent')
            self.order_book.add_order(agent_order)
            
        
        #2 SIMULATE MARKET ACTIVITY FOR CURRENT TIME STEP
        self._simulate_market_orders()
        
        #3. UPDATE AGENT INVENTORY AND CASH BASED ON FILLS
        self._update_agent_state()
        
        #4. ADVANCE TIME STEP
        self.current_step += 1
        self.order_book.set_time(self.current_step)
        
        #5. COMPUTE REWARD
        reward = self._calculate_reward()
        
        #6. CHECK DONE
        terminated = (self.current_step >= self.time_horizon) or (self.inventory <= 0)          # Time limit reached or all shares executed
        truncated = False
    
        
        #7. GET NEXT STATE
        next_state = self._get_state()
        
        #8 INFO DICTIONARY
        info = {
            'inventory': self.inventory,
            'cash_spent': self.cash_spent,
            'shares_executed': self.shares_executed,    
            'current_step': self.current_step
        }
        
        
        return next_state, reward, terminated, truncated, info
    
    # HELPER FUNCTIONS FOR STEP METHOD
    # Simulate Market Orders and Update Order Book
    def _simulate_market_orders(self):
        """Simulate Random Market participants submitting orders, provide liquidity / realistic dynamics for environment
        """
        
        current_price = self.price_series[self.current_step]
        num_orders = np.random.randint(2,6)  # Random number of market orders this step (2 to 5)
        for _ in range(num_orders):
            order = generate_random_order(current_price=current_price)
            self.order_book.add_order(order)
            
    def _update_agent_state(self):
        """Update agent's inventory and cash based on executed orders in the order book
        """
        for order in self.order_book.filled_orders:
            if order.source == 'rl_agent' and order.status == 'filled':
                fill_qty = order.quantity
                fill_price = order.fill_price
                
                # CLAMP FILL QTY TO REMAINING INVENTORY
                fill_qty = min(fill_qty, self.inventory)
                
                # UPDATE TRACKING
                self.inventory -= fill_qty
                self.cash_spent += fill_qty * fill_price
                self.shares_executed += fill_qty
                self.execution_prices.append(fill_price)
                
                # RECORD FILL
                self.agent_fill_history.append({
                    'step': self.current_step,
                    'quantity': fill_qty,
                    'price': fill_price
                })
                
                # SAFETY
                if self.inventory < 0:
                    self.inventory = 0


    # Calculate Reward (e.g., negative execution cost, slippage, etc.)
    def _calculate_reward(self):
        """Calculate reward based on execution performance
        
        Initial reward: Negative slippage from benchmark (VWAP)
        
        Returns:
            reward (float): calculated reward for current step
        """
        
        # No execution yet, small negative reward to encourage action
        if self.shares_executed == 0:
            return -0.01
        
        # BOUNDS CHECK (CLAMP STEP TO VALID RANGE)
        valid_step = min(self.current_step, len(self.price_series) - 1)

        # VWAP Benchmark
        vwap_benchmark = np.mean(self.price_series[:valid_step+1])
        
        # AGENT AVERAGE EXECUTION PRICE
        avg_exec_price = self.cash_spent / self.shares_executed
        
        # SLIPPAGE NORMALISED (DIFFERENCE FROM VWAP)
        slippage = (avg_exec_price - vwap_benchmark) / vwap_benchmark
        
        # REWARD = NEGATIVE SLIPPAGE (SCALE FOR BETTER LEARNING)
        reward = -slippage * 100        
        
        # BONUS REWARD FOR COMPLETING ORDER
        if self.inventory == 0:
            reward += 1.0  
        
        # PENALTY FOR NOT COMPLETING ORDER BY END
        if self.current_step == self.time_horizon - 1 and self.inventory > 0:
            penalty = (self.inventory / self.parent_order_size) * 10.0 
            reward -= penalty
        
        return reward