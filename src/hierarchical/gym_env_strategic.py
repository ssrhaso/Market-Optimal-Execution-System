"""
GYM ENV FOR PPO AGENT

Wraps DQN Tactical Execution layer and provides high level interface for PPO to make strategic decisions

OBSERVES HIGH LEVEL STATE (9,) INCLUDING PACE GUIDANCE
CHOOSES PACE (SLOW, MEDIUM, FAST) AS ACTION
TELLS DQN TACTICAL LAYER THE PACE GUIDANCE

PPO observes 5 dim state,
chooses pace (3 dim action),
Env sets pace, runs tactical DQN layer for N steps,
Aggregates reward over N steps,
Returns to PPO as new state and reward
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hierarchical.gym_env_tactical_guided import TacticalEnvGuided
from stable_baselines3 import DQN

class StrategicEnv(gym.Env):    # LOAD DQN TACTICAL POLICY
    def __init__(
        self,
        tactical_policy = None,
        tactical_steps = 10,
        parent_order_size = 1000,
        time_horizon = 100,
        n_levels = 10
    ):
        
        
        # INITIALISE STRATEGIC ENV
        super().__init__()
    
        # DEFINE OBSERVATION SPACE
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (5,), # STRATEGIC OBSERVATION SPACE (MARKET VOL, BID ASK SPREAD, TIME PROGRESS, INVENTORY % REMAINING, CUMULATIVE REWARD)
            dtype = np.float32
        )
        
        # DEFINE ACTION SPACE (PACE CHOICES)
        self.action_space = spaces.Discrete(3)      # 0 = SLOW, 1 = MEDIUM, 2 = FAST
        
        # DQN EXECUTION ENV
        self.tactical_env = TacticalEnvGuided(
            parent_order_size = parent_order_size,
            time_horizon = time_horizon,
            n_levels = n_levels
        )
        
        
        # STORE REFERENCE AND TACTICAL POLICY
    
        self.tactical_policy = tactical_policy              # PRE-TRAINED DQN POLICY
        self.tactical_steps = tactical_steps                # NUMBER OF TACTICAL STEPS PER STRATEGIC STEP

        # MAP PPO ACTION TO PACE GUIDANCE
        self.pace_targets = {
            0: 0.02,   # SLOW (2%)
            1: 0.05,   # MEDIUM (5%)
            2: 0.10    # FAST (10%)
        }
        
        # TRACKING VARIABLES
        self.cumulative_reward = 0.0    # CUMULATIVE REWARD TRACKER
        self.tactical_obs = None        # CURRENT TACTICAL OBSERVATION
        
        
    def reset(self, seed=None, options=None):
        """
        Reset strategic and tactical envs for new episode
        """
        
        # RESET TACTICAL ENV
        self.tactical_obs, _ = self.tactical_env.reset(seed=seed, options=options)
        
        # RESET CUMULATIVE REWARD
        self.cumulative_reward = 0.0
        
        # RETURN STRATEGIC OBSERVATION, EMPTY DICT  
        return self._get_strategic_state(), {}
    
    
    def step(self, action):
        """
        Execute 1 strategic step
        """
        
        # CONVENRT PPO ACTION (0,1,2) TO PACE TARGET (0.02,0.05,0.10)
        action = int(np.asarray(action).squeeze())
        pace = self.pace_targets[action]


        # SET PACE GUIDANCE IN TACTICAL ENV (UPDATES 9th DIM OF DQN OBS)
        self.tactical_env.set_pace_guidance(pace)
        
        # RUN TACTICAL ENV FOR N STEPS
        reward_sum = 0.0
        tactical_steps_taken = 0
        
        # RUN TACTICAL STEPS
        for _ in range(self.tactical_steps):
            # HANDLE INVENTORY DEPLETION
            if self.tactical_env.inventory <= 0:
                break
            
            # TACTICAL ACTION FROM DQN (ACCORDING TO POLICY)
            if self.tactical_policy is not None:
                tactical_action, _ = self.tactical_policy.predict(
                    self.tactical_obs, 
                    deterministic=True
                )
            else:
                tactical_action = self.tactical_env.action_space.sample()
        
            # EXECUTE IN MARKET
            self.tactical_obs, reward, terminated, truncated, info = self.tactical_env.step(tactical_action)
            
            # ACCUMULATE REWARD
            reward_sum += reward
            tactical_steps_taken += 1
            
            # CHECK TERMINATION
            if terminated or truncated:
                break
            
        # UPDATE CUMULATIVE REWARD
        self.cumulative_reward += reward_sum
        
        # GET STRATEGIC OBSERVATION
        done = (
            self.tactical_env.current_step >= self.tactical_env.time_horizon or 
            self.tactical_env.inventory <= 0
        )
        
        # RETURN STRATEGIC OBSERVATION, REWARD, DONE, TRUNCATED, INFO TO PPO
        return(
            self._get_strategic_state(),
            reward_sum,
            done,
            False, # TRUNCATED FLAG (NOT USED AT STRATEGIC LEVEL)
            {
                'tactical_steps_taken': tactical_steps_taken,
                'pace_chosen': pace,
                'inventory_remaining': self.tactical_env.inventory,
                'current_step': self.tactical_env.current_step,
            }
        )



    def _get_strategic_state(self):
        """
        Construct 5 dim strategic obs for PPO agent
        """
        # 1. VOLATILITY
        if hasattr(self.tactical_env, 'current_volatility'):
            volatility = self.tactical_env.current_volatility
        else:
            volatility = 0.015 # DEFAULT VALUE IF NOT AVAILABLE
        if np.isnan(volatility):
            volatility = 0.010
        
        
        # 2. BID-ASK SPREAD (NORMALISED)
        best_bid, best_ask = self.tactical_env.order_book.get_top_of_book()
        
        valid_step = min(                                   # Valid index for price series
            self.tactical_env.current_step,
            len(self.tactical_env.price_series) - 1
        )
        
        if hasattr(self.tactical_env, 'price_series') and len(self.tactical_env.price_series) > 0:
            current_price = self.tactical_env.price_series[valid_step]
        else:
            current_price = 100.0 # DEFAULT PRICE IF NOT AVAILABLE
        # SPREAD
        if best_bid is None or best_ask is None or current_price == 0:
            spread = 0.0
        else:
            spread = (best_ask - best_bid) / current_price  # Normalised spread
        if np.isnan(spread):
            spread = 0.0
        
        
        
            
            
        
        # 3. TIME PROGRESS (0 TO 1)
        time_progress = self.tactical_env.current_step / self.tactical_env.time_horizon
        if np.isnan(time_progress):
            time_progress = 0.0
        
        # 4. INVENTORY % REMAINING (0 TO 1)
        inventory_pct = self.tactical_env.inventory / self.tactical_env.parent_order_size 
        if np.isnan(inventory_pct):
            inventory_pct = 1.0
        
        # 5. CUMULATIVE REWARD
        cumulative_reward = self.cumulative_reward
        if np.isnan(cumulative_reward):
            cumulative_reward = 0.0
        
        # STATE ARRAY (5 DIM FOR GYM)
        state = np.array([
            volatility,
            spread,
            time_progress,
            inventory_pct,
            cumulative_reward
        ], dtype=np.float32)
        if np.any(np.isnan(state)):
            print("NaN in strategic state: " , state)
            state = np.nan_to_num(state, nan= 0.0)
            
        return state
    
    
    
    
    
 