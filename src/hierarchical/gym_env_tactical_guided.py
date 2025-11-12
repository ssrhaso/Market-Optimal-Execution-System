"""
GYM EVN FOR DQN AGENT

GYM2 - Tactical Execution Enviroment with Strategic Pace Guidance
Extend previous envionment to accept pace guidance from strategic layer
State space is extended (8 -> 9) to include pace guidance
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_env_baseline import OptimalExecutionEnv


class TacticalEnvGuided(OptimalExecutionEnv):
    """
    Inherits from OptimalExecutionEnv
    - 8 -> 9 state space to include pace guidance
    - Receive pace target via set_pace_guidance()
    - Reward include small bonus for following the suggested pace
    
    Pace Guidance:
    - 0.02 = slow (2% of remaining inv)
    - 0.05 = medium (5%)
    - 0.10 = fast (10%)
    """

    def __init__(self, parent_order_size = 1000, time_horizon = 100, n_levels = 10):
        super().__init__(parent_order_size, time_horizon, n_levels)                     # Initialise base env logic (order book, price sim, etc)
        
        # EXTEND STATE SPACE
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (9,), # EXTENDED STATE SPACE (9,)
            dtype = np.float32
        )
        
        
        # TRACK PACE GUIDANCE
        self.current_pace_target = 0.05 # Default to medium pace
        # TRACK EXECUTION RATE FOR PACE FOLLOWING REWARD
        self.steps_since_pace_set = 0
        self.shares_executed_since_pace_set = 0
        
    # PACE GUIDANCE INTERFACE    
    def set_pace_guidance(self, pace_target):
        """
        Called by strategic layer to set pace guidance for current episode
        pace_target: float (e.g., 0.02, 0.05, 0.10)
        """
        
        self.current_pace_target = pace_target
        self.steps_since_pace_set = 0
        self.shares_executed_since_pace_set = 0
        
    # RESET OVERRIDE
    def reset(self, seed=None, options=None):
        """
        Override to reset pace tracking 
        """
        obs, info = super().reset(seed=seed, options=options)   # Call base reset
        
        # RESET PACE TRACKING
        self.current_pace_target = 0.05     # Default to medium pace
        self.steps_since_pace_set = 0
        self.shares_executed_since_pace_set = 0
        return self._get_state(), info
    
    # STATE OBSERVATION
    def _get_state(self):
        # EXTEND ORIGINAL STATE WITH PACE GUIDANCE
        original_state = super()._get_state()  # Get original 8-dim state
        state_with_pace = np.append(original_state, self.current_pace_target) # Append pace guidance
        
        return state_with_pace.astype(np.float32) # Ensure correct dtype for training
    
    
    # STEP METHOD (EXECUTION LOOP)
    def step(self, action):
        """
        Override to include pace following reward
        """
        # TRACK SHARES BEFORE STEP
        shares_before = self.shares_executed
        
        # EXECUTE STEP USING PARENT LOGIC
        obs, reward, done, truncated, info = super().step(action)
        
        # TRACK SHARES AFTER STEP
        shares_this_step = self.shares_executed - shares_before
        self.shares_executed_since_pace_set += shares_this_step
        self.steps_since_pace_set += 1
        
        # ADD PACE ADHERANCE BONUS TO REWARD
        pace_bonus = self._calculate_pace_adherence_bonus()
        reward += pace_bonus
        
        return self._get_state(), reward, done, truncated, info
    
    
    def _calculate_pace_adherence_bonus(self):
        """Calculate small reward bonus for following pace guidance
        
        - If executing close to pace - small positive bonus
        - If executing too fast or too slow - small negative penalty
        """
        if self.steps_since_pace_set == 0 or self.inventory == 0:
            return 0.0  # No bonus if no steps taken (new pace guidance) or no inventory left
        
        # CALCULATE ACTUAL EXCECUTION RATE
        actual_rate = self.shares_executed_since_pace_set / (       # On average, which fraction of inventory executed per step
            self.steps_since_pace_set * max(self.inventory, 1)      # Avoid division by zero
        )
        
        # TARGET RATE 
        target_rate = self.current_pace_target
        
        # ADHERENCE ERROR (NORMALISED)
        if target_rate > 0:
            adherence_error = abs(actual_rate - target_rate) / target_rate
        else:
            adherence_error = 0.0
        
        # DETERMINE BONUS/PENALTY
        if adherence_error < 0.2:
            pace_bonus = 0.05
        else:
            pace_bonus = -0.01 * min(adherence_error, 1.0)  # Cap penalty for deviation
    
        return pace_bonus
