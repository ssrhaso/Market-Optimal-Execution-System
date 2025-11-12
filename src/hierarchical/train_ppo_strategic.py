"""
Trains PPO agent for strategic layer using the StrategicEnv environment

2 stage training pipeline

1. Train PPO with RANDOM tactical layer (DQN untrained)
- Fast, PPO learns basic strategy without DQN knowledge, Validates PPO setup

2. Train PPO with PRETRAINED tactical layer (DQN pretrained)
- Full hierarchical system, PPO learns to leverage DQN skills, produces final model
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import random
import torch
from stable_baselines3 import PPO, DQN
from hierarchical.gym_env_strategic import StrategicEnv

GLOBAL_SEED = 42
def set_global_seed(seed):
    """
    Set seeds for reproducibility across numpy, random, and torch
    """
    np.random.seed(seed)            # GBM simulator, order book generation, env sampling
    random.seed(seed)               # random.randint, random.choice() etc.
    torch.manual_seed(seed)         # PPO network initialisation, gradient updates
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # GPU operations must also be seeded
    torch.backends.cudnn.deterministic = True   # Deterministic convolution algorithms (for same speed)
    torch.backends.cudnn.benchmark = False

def train_ppo_stage1():
    """
    S1: Train PPO with RANDOM tactical layer execution (MVP setup)
    
    - CREATE StrategicEnv with UNTRAINED DQN tactical policy
    - INITIALISE PPO AGENT
    - TRAIN FOR 50K TIMESTEPS
    - SAVE PPO MODEL
    """
    # SETUP / LOGGING
    print("\n[S1] Training PPO with RANDOM tactical layer...")
    os.makedirs("models/ppo_strategic", exist_ok=True)
    print("\n[SETUP] DIRECTORY : models/ppo_strategic")
    print("[SETUP] SEED: {}".format(GLOBAL_SEED))
    
    #STRATEGIC ENV SETUP (PPO ENV)
    env = StrategicEnv(
        tactical_policy = None,          # RANDOM TACTICAL POLICY (UNTRAINED DQN)
        tactical_steps = 10,             # MAKE DECISION EVERY 10 TACTICAL STEPS
        parent_order_size = 1000,        # TOTAL SHARES
        time_horizon = 100,              # TOTAL TIME STEPS (MINUTES)
        n_levels = 10                    # ORDER BOOK DEPTH
    )
    
    # DEBUG OUTPUT
    print(f"  ✓ Observation space shape: {env.observation_space.shape}")
    print(f"    → PPO observes 5-dim state: [volatility, spread, time%, inventory%, reward_sum]")
    print(f"  ✓ Action space: {env.action_space}")
    print(f"    → PPO chooses: 0=slow (2%), 1=medium (5%), 2=fast (10%)")
    print(f"  ✓ Tactical steps per strategic decision: {env.tactical_steps}")
    print(f"    → After PPO chooses pace, DQN executes for 10 steps, then PPO decides again")
    
    
    # INITIALISE PPO AGENT
    print("\n[S1] Initialising PPO agent...")
    
    model = PPO(
        policy="MlpPolicy",                 # Multi-layer perceptron policy (feedforward NN)
        env=env,                            # Environment
        learning_rate=3e-4,                 # Learning rate (0.0003 standard for PPO)
        n_steps= 128,                        # Number of steps to run per environment per update (128 steps before each learning update)
        
        # GRADIENT PARAMETERS
        batch_size=32,                      # Number of experiences to sample for each learning update (32 experiences at a time)
        n_epochs = 10,                      # Optimise each batch 10 times before collecting new data

        # RL SPECIFIC PARAMETERS
        gamma=0.99,                         # Discount factor for future rewards (future rewards worth 99% of immediate rewards)
        gae_lambda=0.95,                    # How much to trust value function estimates (0 = no trust, 1 = full trust)
            
        # TRUST REGION PARAMETERS
        clip_range=0.2,                     # Clip range for PPO surrogate objective (prevents large policy updates) 0.2 = 20% change max

        seed = GLOBAL_SEED,                 # Global seed for reproducibility
     
        verbose=1,                          # Verbosity level (1 = info messages at each step)
        tensorboard_log="./ppo_tensorboard_stage1/" # Tensorboard log directory
        )
    
           
    print("✓ PPO Agent initialised")
    print(f"  - Policy network: {model.policy}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Discount factor (gamma): {model.gamma}")
    
    # TRAIN PPO AGENT
    print("\n[S1] Training PPO Agent...")
    print("(This may take 5 - 10 minutes)")
    
    total_timesteps = 50000                     # Total training timesteps (50K for stage 1)
    model.learn(
        total_timesteps=total_timesteps,        # Total training timesteps
        log_interval=env.tactical_steps,        # Log every 10 steps 
        progress_bar=True                       # Progress bar display
    )
    print("✓ Training complete")
    # SAVE PPO MODEL
    print("\n[S1] Saving trained PPO model...")
    model_path= "models/ppo_strategic/ppo_strategic_stage1_random_tactical"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}.zip") # Will save data, pytorch_variables (weights), sb3 version info all zipped
    print("✓ Stage 1 complete")
    return model


def train_ppo_stage2():
    """
    S2: Train PPO with PRETRAINED DQN tactical layer execution (FULL HIERARCHICAL SETUP)
    
    - LOAD PRETRAINED DQN TACTICAL POLICY
    - CREATE StrategicEnv with PRETRAINED DQN tactical policy
    - INITIALISE PPO AGENT
    - TRAIN FOR 100K TIMESTEPS
    - SAVE PPO MODEL
    """
    # SETUP / LOGGING
    print("\n[S2] Training PPO with PRETRAINED DQN tactical layer...")
    
    # LOAD PRETRAINED DQN TACTICAL POLICY
    print("\n[SETUP] Loading pretrained DQN Model...")
    try:
        dqn_model = DQN.load("models/dqn_guided/dqn_guided")   # Load pretrained DQN guided model
        print("✓ Pretrained DQN model loaded")
        print(f"  - Model: {dqn_model}")
        print("  - DQN has been trained on tactical execution with pace guidance")
    except FileNotFoundError:
        print("✗ Pretrained DQN model not found. Please train DQN tactical model first.")
        print("\n ACTION REQUIRED: ")
        print("  1. Run 'train_dqn_guided.py' to train the DQN tactical model with pace guidance.")
        print("  2. Expected output for model: 'models/dqn_guided/dqn_guided.zip'.")
        print("  3. After obtaining model, re-run this script to train the PPO strategic model.")
        return None
    
    #STRATEGIC ENV SETUP (PPO ENV)
    env = StrategicEnv(
        tactical_policy = dqn_model,     # PRETRAINED DQN TACTICAL POLICY
        tactical_steps = 10,             # MAKE DECISION EVERY 10 TACTICAL STEPS
        parent_order_size = 1000,        # TOTAL SHARES
        time_horizon = 100,              # TOTAL TIME STEPS (MINUTES)
        n_levels = 10                    # ORDER BOOK DEPTH
    )
    
    print(f"  ✓ Observation space shape: {env.observation_space.shape}")
    print(f"    → PPO observes 5-dim state: [volatility, spread, time%, inventory%, reward_sum]")
    print(f"  ✓ Action space: {env.action_space}")
    print(f"    → PPO chooses: 0=slow (2%), 1=medium (5%), 2=fast (10%)")
    print(f"  ✓ Tactical steps per strategic decision: {env.tactical_steps}")
    print(f"    → After PPO chooses pace, DQN executes for 10 steps, then PPO decides again")
    
    # INITIALISE PPO AGENT
    print("\n[S2] Initialising PPO agent...")
    model = PPO(
        policy="MlpPolicy",                 # Multi-layer perceptron policy (feedforward NN)
        env=env,                            # Environment
        
        learning_rate=1e-4,                 # Learning rate (LOWER FOR FINE TUNING)

        # SAME PARAMTERS AS S1
        batch_size=32,                      # Number of experiences to sample for each learning update (32 experiences at a time)
        n_epochs = 10,                      # Optimise each batch 10 times before collecting new data
        gamma=0.99,                         # Discount factor for future rewards (future rewards worth 99% of immediate rewards)
        gae_lambda=0.95,                    # How much to trust value function estimates (0 = no trust, 1 = full trust)
        clip_range=0.2,                     # Clip range for PPO surrogate objective (prevents large policy updates) 0.2 = 20% change max
        seed = GLOBAL_SEED,                 # Global seed for reproducibility
        verbose=1,                          # Verbosity level (1 = info messages at each step)
        tensorboard_log="./ppo_tensorboard_stage2/" # Tensorboard log directory)
        )  
    
           
    print("✓ PPO Agent initialised")
    print(f"  - Policy network: {model.policy}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Discount factor (gamma): {model.gamma}")
    
    # TRAIN PPO AGENT
    print("\n[S2] Training PPO Agent...")
    print("(This may take 10 - 15 minutes)")
    print("Monitor Progress: tensorboard --logdir ./ppo_tensorboard_stage2/")
    
      
    total_timesteps = 100000                     # Total training timesteps (100K for stage 2) 
    model.learn(
        total_timesteps=total_timesteps,        # Total training timesteps
        log_interval=env.tactical_steps,        # Log every 10 steps 
        progress_bar=True                       # Progress bar display
    )
    print("✓ Training complete")
    # SAVE PPO MODEL
    print("\n[S2] Saving trained PPO model...")
    model_path= "models/ppo_strategic/ppo_strategic_stage2_random_tactical"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}.zip") # Will save data, pytorch_variables (weights), sb3 version info all zipped
    print("✓ Stage 2 complete")
    return model
    
    
    
# MAIN
def main():
    """
    Main entry point: orchestrates both training stages sequentially.
    
    FLOW:
    1. Set global seed for reproducibility
    2. Run Stage 1 (PPO with random tactics)
    3. Run Stage 2 (PPO with trained DQN)
    4. Print next steps
    """
    
    # Set all random seeds BEFORE any randomness happens
    set_global_seed(GLOBAL_SEED)
    
    print("\n" + "=" * 80)
    print("PPO STRATEGIC AGENT TRAINING PIPELINE")
    print("=" * 80)
    print("\nThis script trains the high-level (strategic) PPO agent.")
    print("PPO learns which execution pace maximizes trading performance.")
    print("\nTraining occurs in TWO STAGES for validation and reproducibility.")
    print(f"Global seed: {GLOBAL_SEED} (all runs are reproducible)")
    print("=" * 80)
    
    # RUN STAGE 1
        
    ppo_stage1 = train_ppo_stage1()
    
    # Check if training succeeded (non-None means success)
    if ppo_stage1 is None:
        print("\n✗ Stage 1 training failed")
        print("  Check error messages above")
        return
    
    # Print success banner
    print("\n" + "=" * 80)
    print("✓ STAGE 1 COMPLETE: PPO trained with random tactical execution")
    print("=" * 80)
    print("  → Proof of concept: hierarchical system works")
    print("  → Model saved: models/ppo_strategic/ppo_stage1_random_tactical.zip")
    
    # RUN STAGE 2
    
    ppo_stage2 = train_ppo_stage2()
    
    # Check if training succeeded
    if ppo_stage2 is None:
        print("\n✗ Stage 2 training failed")
        print("  Make sure you have trained DQN at: models/dqn_guided/dqn_guided.zip")
        print("  Run train_dqn_guided.py first if you haven't already")
        return
    
    # Print success banner
    print("\n" + "=" * 80)
    print("✓ STAGE 2 COMPLETE: PPO trained with trained DQN (FULL HIERARCHY)")
    print("=" * 80)
    print("  → Full hierarchical RL system complete")
    print("  → PPO coordinates with trained DQN")
    print("  → Model saved: models/ppo_strategic/ppo_stage2_with_dqn.zip")
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("✓✓✓ PPO TRAINING PIPELINE COMPLETE ✓✓✓")
    print("=" * 80)
    
    print("\n[REPRODUCIBILITY]")
    print(f"   All runs use seed={GLOBAL_SEED}")
    print("   To reproduce: run this script again (same results)")
    print("   To stress-test: try different seeds (e.g., 123, 456)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
        
    
    
    
    
   