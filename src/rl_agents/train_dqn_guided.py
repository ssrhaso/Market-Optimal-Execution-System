"""Train a Deep Q-Network (DQN) agent for the Optimal Execution environment 

ADJUSTED USING PACE GUIDANCE.

1. Create execution environment
2. Initialise DQN agent with NN architecture
3. Train agent over multiple timesteps (episodes)
4. Saves trained model
5. Evaluate agent performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from hierarchical.gym_env_tactical_guided import TacticalEnvGuided

# SET ALL SEEDS FOR REPRODUCIBILITY
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



class TrainingCallback(BaseCallback):
    """Custom callback to log training progress & metrics using Stable Baselines3 custom callback system"""
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)         # Initiliase callback
        self.episode_rewards = []                           
        self.episode_lengths = []                    
        self.current_episode_reward = 0
        self.current_episode_length = 0       
        
    
    # ON STEP FUNCTION - CALLED AT EACH STEP DURING TRAINING
    def _on_step(self) -> bool:
        """Informs the callback the status of the current step in training"""
        
        self.current_episode_reward += self.locals.get('rewards')[0]   # Accumulate reward for current episode
        self.current_episode_length += 1                               # Accumulate length for current episode
        
        
        # CHECK IF EPISODE FINISHED
        if self.locals.get('dones')[0] == True:                     # If done flag is True (episode finished)
            self.episode_rewards.append(self.current_episode_reward)   # Log episode reward
            self.episode_lengths.append(self.current_episode_length)   # Log episode length
    
            # PRINT METRICS EVERY 10 EPISODES
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])      # Average reward over last 10 episodes
                print(f"Episode {len(self.episode_rewards)} - Average Reward (last 10): {avg_reward:.2f}")
            
            # RESET CURRENT EPISODE TRACKERS
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        # CONTINUE TRAINING
        return True
        


# Main training function

def train_dqn_agent():
    """MAIN TRAINING FUNCTION"""
    
    print("=" * 60)
    print("TRAINING DQN AGENT FOR OPTIMAL EXECUTION ENVIRONMENT")
    print("=" * 60)
    
    
    # 1. CREATE ENV
    
    print("\n[1] Creating Optimal Execution Environment...")
    env = TacticalEnvGuided(
        parent_order_size=1000,     # Total shares to execute
        time_horizon=100,           # Total time steps in the episode
        n_levels=10)                # Number of price levels in the order book
    env.reset(seed=42)              # Reset environment with seed for reproducibility
    
    
    print("✓ Environment created")
    print(f"  - Action space: {env.action_space}")              # Features of action space
    print(f"  - Observation space: {env.observation_space}")    # Possible actions
    
    #2. INITIALISE DQN AGENT
    print("\n[2] Initialising DQN Agent...")
    
    model = DQN(
        policy="MlpPolicy",                 # Multi-layer perceptron policy (feedforward NN)
        env=env,                            # Environment
        learning_rate=5e-5,                 # Learning rate (how fast the agent learns)
        buffer_size=10000,                  # How many past experiences to store for learning
        learning_starts=1000,               # Number of steps before learning starts (to gather initial experience)
        batch_size=32,                      # Number of experiences to sample for each learning update (32 experiences at a time)
        tau=1.0,                            # Target network update rate (1.0 means hard update)
        gamma=0.99,                         # Discount factor for future rewards (future rewards worth 99% of immediate rewards)
        train_freq=4,                       # Frequency of training updates (every 4 steps)
        gradient_steps=1,                   # Number of gradient steps per training update
        target_update_interval=1000,        # How often to update the target network (every 1000 steps)
        exploration_fraction=0.3,           # Fraction of total training steps for exploration (first 30% of training spent exploring)
        exploration_initial_eps=1.0,        # Initial exploration probability (start fully exploring) (100% random actions during exploration)
        exploration_final_eps=0.05,         # Final exploration probability (end with 5% random actions)
        verbose=1,                          # Verbosity level (1 = info messages)
        tensorboard_log="./dqn_guided_tensorboard/" # Tensorboard log directory)
        )  
        
    print("✓ DQN Agent initialised")
    print(f"  - Policy network: {model.policy}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Replay Buffer Size: {model.replay_buffer.buffer_size}")
    
    # 3. TRAIN AGENT
    print("\n[3] Training DQN Agent...")
    print("(This may take 5 - 10 minutes)")
    
    # CALLBACK
    callback = TrainingCallback()               #  Create custom callback to log training progress
    # TIMESTEPS TO TRAIN
    total_timesteps = 100000                     # Total training timesteps (can be adjusted)
    
    model.learn(
        total_timesteps=total_timesteps,        # Total training timesteps
        callback=callback,                      # Custom callback for logging
        log_interval=100,                       # Log every 100 steps 
        progress_bar=True                       # Progress bar display
    )
    print("✓ Training complete")
    
    
    # 4. SAVE TRAINED MODEL
    print("\n[4] Saving trained model...")
    os.makedirs("models/dqn_guided", exist_ok=True)          # Create directory if it doesn't exist
    model_path= "models/dqn_guided/dqn_guided"    
    model.save(model_path)
    print(f"✓ Model saved to {model_path}") # Will save data, pytorch_variables (weights), sb3 version info all zipped
    
    
    # 5. EVALUATE AGENT
    print("\n[5] Evaluating trained agent...")
    
    # SETUP EVALUATION ENV (for 10 episodes)
    num_eval_episodes = 10
    eval_rewards = []
    eval_inventories = []
    eval_cash_spent = []
    
    for i in range(num_eval_episodes):
        
        obs, info = env.reset()                 # Reset environment for new episode (initial observation) (unpack tuple)
        episode_reward = 0                      # Initialise episode reward
        done = False                            # Done flag
    
        # RUN EPISODE UNTIL DONE
        while not done:
            # USE TRAINED MODEL TO "PREDICT" ACTIONS
            action, _states = model.predict(obs, deterministic=True)    # Get action from trained model, deterministic, no exploration (random actions). Ignore states.
            
            obs, reward, terminated, truncated, info = env.step(action)                  # Take action in environment (step function)
            episode_reward += reward                                                     # Accumulate reward for the episode
            done = terminated or truncated                                               # Update done flag
        
        eval_rewards.append(episode_reward)                           # Store reward for episode
        eval_inventories.append(info['inventory'])                    # Store final inventory (should be 0 if fully executed)
        eval_cash_spent.append(info['cash_spent'])                    # Store cash spent 
            
        # PRINT EVALUATION METRICS
        print(f"Episode {i+1}: Reward: {episode_reward:.2f}, "
              f"Final Inventory: {info['inventory']}, "
              f"Cash Spent: {info['cash_spent']:.2f}")
        
    # PRINT AVERAGE EVALUATION METRICS
    print(f"\nAverage Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average Final Inventory: {np.mean(eval_inventories):.1f}")
    print(f"Average Cash Spent: {np.mean(eval_cash_spent):.2f}")
    print("\n✓ Evaluation complete")
    
    
    # 6. VISUALISATION - PLOT TRAINING REWARD PROGRESS
    print("\n[6] Plotting training reward progress...")
    
    # PLOT REWARD PROGRESS IF AVAILABLE
    if len(callback.episode_rewards) > 0:
        
        # INITIALISE PLOT
        plt.figure(figsize=(10, 6)) 
        window = 10
        if len(callback.episode_rewards) >= window:
            smoothed = np.convolve(                     # Convolve creates moving average for smoothing reward data (better visualisation)
                callback.episode_rewards, 
                np.ones(window)/window, 
                mode='valid'    
            )
            plt.plot(smoothed, label='Smoothed Reward', color='lightgreen', linewidth=2)
        
        # PLOT RAW EPISODE REWARDS (30% TRANSPARENCY)
        plt.plot(callback.episode_rewards, label='Episode Reward', color='lightblue', alpha=0.3)
        plt.title('DQN Training Reward Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SAVE AND DISPLAY PLOT (RESOLUTION 150 DPI, SAVED TO models directory)
        plt.savefig("models/dqn_execution/training_curve.png", dpi=150, bbox_inches='tight')
        print("✓ Training reward plot saved to models/dqn_execution/training_curve.png")
        plt.show()
        
        
        
    print("\n" + "=" * 60)
    print("DQN AGENT TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    return model           



if __name__ == "__main__":
    trained_model = train_dqn_agent()
    
    
        