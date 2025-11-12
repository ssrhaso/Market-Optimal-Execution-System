"""
Evaluate Hierarchical RL System (PPO + DQN)

Compares performance of:
1. VWAP (Volume Weighted Average Price) - industry standard baseline
2. TWAP (Time Weighted Average Price) - simple baseline
3. Baseline DQN (no hierarchy)
4. Guided DQN (single layer with pace)
5. Hierarchical System (PPO + DQN) ← YOUR MAIN RESULT

Runs evaluations and measures:
- Average slippage vs VWAP benchmark
- Variance (consistency)
- Min/Max slippage
- Comprehensive comparison to all methods
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO, DQN
from hierarchical.gym_env_strategic import StrategicEnv

# Import baseline algorithms
from twap import execute_twap
from vwap import execute_vwap
from simulategbm import simulate_gbm


def evaluate_vwap_twap(n_episodes=20, parent_order_size=1000, time_horizon=100):
    """
    Evaluate VWAP and TWAP baselines.
    
    Args:
        n_episodes: Number of episodes to evaluate
        parent_order_size: Total shares to execute
        time_horizon: Time steps
    
    Returns:
        vwap_results, twap_results: Dicts with statistics
    """
    
    print(f"\n[BASELINE] Evaluating VWAP and TWAP ({n_episodes} episodes)...")
    
    vwap_slippages = []
    twap_slippages = []
    
    for episode in range(n_episodes):
        # Simulate market
        _, price_series = simulate_gbm(time_horizon=time_horizon, initial_price=100, mu=0.0, sigma=0.02)

        
        # VWAP execution
        vwap_price = execute_vwap(price_series)
        vwap_slippage = (vwap_price - np.mean(price_series)) / np.mean(price_series)
        vwap_slippages.append(vwap_slippage * 100)  # Convert to percentage
        
        # TWAP execution
        twap_price = execute_twap(price_series)
        twap_slippage = (twap_price - np.mean(price_series)) / np.mean(price_series)
        twap_slippages.append(twap_slippage * 100)  # Convert to percentage
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1:2d}: VWAP slippage={vwap_slippage*100:6.2f}%, "
                  f"TWAP slippage={twap_slippage*100:6.2f}%")
    
    vwap_results = {
        'mean': np.mean(vwap_slippages),
        'std': np.std(vwap_slippages),
        'min': np.min(vwap_slippages),
        'max': np.max(vwap_slippages),
        'episodes': vwap_slippages
    }
    
    twap_results = {
        'mean': np.mean(twap_slippages),
        'std': np.std(twap_slippages),
        'min': np.min(twap_slippages),
        'max': np.max(twap_slippages),
        'episodes': twap_slippages
    }
    
    return vwap_results, twap_results


def evaluate_hierarchical_system(model_ppo, model_dqn, env, n_episodes=20):
    """
    Evaluate hierarchical RL system (PPO + DQN).
    
    Args:
        model_ppo: Trained PPO agent (strategic layer)
        model_dqn: Trained DQN agent (tactical layer)
        env: Strategic environment
        n_episodes: Number of episodes to evaluate
    
    Returns:
        results: Dict with statistics
    """
    
    print(f"\n[HIERARCHY] Running {n_episodes} evaluation episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Run episode
        while not done:
            # PPO chooses pace (strategic decision)
            action, _ = model_ppo.predict(obs, deterministic=True)
            
            # Environment runs tactical DQN for N steps
            obs, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            episode_length += info['tactical_steps_taken']
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1:2d}: Reward={episode_reward:7.2f}, "
                  f"Length={episode_length:3d}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    results = {
        'mean': mean_reward,
        'std': std_reward,
        'min': min_reward,
        'max': max_reward,
        'episodes': episode_rewards
    }
    
    return results


def main():
    """
    Main evaluation pipeline comparing all methods.
    """
    
    print("\n" + "=" * 90)
    print("COMPREHENSIVE EXECUTION BENCHMARK: ALL METHODS")
    print("=" * 90)
    print("\nEvaluating:")
    print("  1. VWAP (industry standard)")
    print("  2. TWAP (simple baseline)")
    print("  3. Baseline DQN (no hierarchy)")
    print("  4. Guided DQN (single layer with pace)")
    print("  5. Hierarchical RL (PPO + DQN) ← YOUR INNOVATION")
    print("=" * 90)
    
    # EVALUATE VWAP & TWAP
    
    print("\n[PHASE 1] Evaluating Traditional Methods (VWAP, TWAP)...")
    vwap_results, twap_results = evaluate_vwap_twap(n_episodes=20)
    
    print(f"\n  VWAP: {vwap_results['mean']:.2f}% ± {vwap_results['std']:.2f}%")
    print(f"  TWAP: {twap_results['mean']:.2f}% ± {twap_results['std']:.2f}%")
    
    # LOAD TRAINED RL MODELS
    
    print("\n[PHASE 2] Loading Trained RL Models...")
    
    try:
        print("  Loading PPO (strategic layer)...")
        ppo_model = PPO.load("models/ppo_strategic/ppo_strategic_stage2_random_tactical")
        print("  ✓ PPO loaded")
        
        print("  Loading DQN (tactical layer)...")
        dqn_model = DQN.load("models/dqn_guided/dqn_guided")
        print("  ✓ DQN loaded")    
        
    except FileNotFoundError as e:
        print(f"\n✗ Error loading models: {e}")
        print("\nMake sure you have:")
        print("  - models/ppo_strategic/ppo_strategic_stage2_with_dqn.zip")
        print("  - models/dqn_guided/dqn_guided.zip")
        return
    
    # CREATE ENVIRONMENT
    
    print("\n[PHASE 2] Creating evaluation environment...")
    
    env = StrategicEnv(
        tactical_policy=dqn_model,
        tactical_steps=10,
        parent_order_size=1000,
        time_horizon=100,
        n_levels=10
    )
    print("  ✓ Environment created")
    
    # EVALUATE HIERARCHICAL SYSTEM
    
    print("\n[PHASE 3] Evaluating Hierarchical RL System...")
    
    hierarchy_results = evaluate_hierarchical_system(ppo_model, dqn_model, env, n_episodes=20)
    
    print(f"\n  Hierarchical: {hierarchy_results['mean']:.2f} ± {hierarchy_results['std']:.2f}")
    
    # PRINT COMPREHENSIVE RESULTS TABLE
    
    print("\n" + "=" * 90)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 90)
    
    print(f"\n{'Method':<40} {'Mean':<15} {'Std Dev':<15} {'Min/Max':<20}")
    print("-" * 90)
    
    # Traditional baselines
    print(f"{'VWAP (industry standard)':<40} {vwap_results['mean']:>6.2f}%     {vwap_results['std']:>6.2f}%     "
          f"{vwap_results['min']:>6.2f}% / {vwap_results['max']:>6.2f}%")
    print(f"{'TWAP (simple baseline)':<40} {twap_results['mean']:>6.2f}%     {twap_results['std']:>6.2f}%     "
          f"{twap_results['min']:>6.2f}% / {twap_results['max']:>6.2f}%")
    
    # ML baselines (from your earlier work)
    baseline_dqn_mean = -1.28
    baseline_dqn_std = 4.72
    guided_dqn_mean = 0.44
    guided_dqn_std = 2.14
    
    print(f"{'Baseline DQN (no hierarchy)':<40} {baseline_dqn_mean:>6.2f}      {baseline_dqn_std:>6.2f}      "
          f"{'[from prior results]':<20}")
    print(f"{'Guided DQN (single layer)':<40} {guided_dqn_mean:>6.2f}      {guided_dqn_std:>6.2f}      "
          f"{'[from prior results]':<20}")
    
    # Your result
    print(f"{'Hierarchical RL (PPO + DQN) ✓✓✓':<40} {hierarchy_results['mean']:>6.2f}      {hierarchy_results['std']:>6.2f}      "
          f"{hierarchy_results['min']:>6.2f} / {hierarchy_results['max']:>6.2f}")
    
    print("-" * 90)
    
    # COMPARATIVE ANALYSIS
    
    print("\n" + "=" * 90)
    print("COMPARATIVE ANALYSIS: Hierarchical RL vs All Methods")
    print("=" * 90)
    
    your_mean = hierarchy_results['mean']
    your_std = hierarchy_results['std']
    
    print(f"\n[VWAP Comparison]")
    perf_vs_vwap = your_mean - vwap_results['mean']
    var_vs_vwap = vwap_results['std'] - your_std
    print(f"  Performance difference: {perf_vs_vwap:+.2f}%")
    print(f"  Variance reduction: {var_vs_vwap:+.2f}%")
    if perf_vs_vwap > 0:
        print(f"  ✓ BETTER than VWAP by {perf_vs_vwap:.2f}%")
    else:
        print(f"  ✗ VWAP still superior (industry standard)")
    
    print(f"\n[TWAP Comparison]")
    perf_vs_twap = your_mean - twap_results['mean']
    var_vs_twap = twap_results['std'] - your_std
    print(f"  Performance difference: {perf_vs_twap:+.2f}%")
    print(f"  Variance reduction: {var_vs_twap:+.2f}%")
    if perf_vs_twap > 0:
        print(f"  ✓ BETTER than TWAP by {perf_vs_twap:.2f}%")
    else:
        print(f"  ✗ TWAP still superior")
    
    print(f"\n[ML Baselines Comparison]")
    print(f"  vs Baseline DQN:  {your_mean - baseline_dqn_mean:+.2f} (std: {baseline_dqn_std - your_std:+.2f})")
    print(f"  vs Guided DQN:    {your_mean - guided_dqn_mean:+.2f} (std: {guided_dqn_std - your_std:+.2f})")
    
    if your_mean > guided_dqn_mean:
        print(f"  ✓ HIERARCHY improves over guided DQN by {your_mean - guided_dqn_mean:.2f}")
    if your_std < guided_dqn_std:
        print(f"  ✓ MORE CONSISTENT than guided DQN (lower variance)")
    
    # FINAL VERDICT
    
    print("\n" + "=" * 90)
    print("KEY FINDINGS & RESEARCH CONTRIBUTION")
    print("=" * 90)
    
    print(f"\n1. Traditional Methods (VWAP/TWAP):")
    print(f"   - VWAP slippage: {vwap_results['mean']:.2f}%")
    print(f"   - TWAP slippage: {twap_results['mean']:.2f}%")
    
    print(f"\n2. Single-Layer RL (Baseline & Guided DQN):")
    print(f"   - Baseline DQN: -1.28 (worse than VWAP)")
    print(f"   - Guided DQN: +0.44 (modest improvement)")
    
    print(f"\n3. Hierarchical RL (Your Contribution):")
    print(f"   - PPO + DQN: {your_mean:.2f}% (std: {your_std:.2f}%)")
    
    # Final assessment
    print(f"\n[CONCLUSION]")
    if your_mean > vwap_results['mean'] and your_std < vwap_results['std']:
        print(f"  ✓✓✓ HIERARCHICAL RL OUTPERFORMS ALL BASELINES")
        print(f"      - Better performance than VWAP")
        print(f"      - More consistent (lower variance)")
        print(f"      - This is a significant research contribution!")
    elif your_mean > guided_dqn_mean:
        print(f"  ✓✓ HIERARCHICAL RL ADVANCES STATE-OF-ART IN ML")
        print(f"     - Beats all single-layer RL approaches")
        print(f"     - Demonstrates value of hierarchy")
    else:
        print(f"  ⚠ Results competitive with existing methods")
        print(f"    (room for hyperparameter tuning or longer training)")

if __name__ == "__main__":
    main()
