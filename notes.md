# RESEARCH LOG

## DQN BASELINE 

### Functional DQN Baseline, single order, intraday VWAP execution

- Custom Gym env (execution_env.py)
- State (8 features) = normalised - best bid, best ask, spread, time progress, inventory remaining, current volatility, execution progress, avg execution price.
- Action (3 discrete) = buy, hold, sell
- Reward (negative slippage vs VWAP, completion bonus)

DQN agent with experience replay
Realistic market simulator (order book, gbm price, random orders)

### RESULTS:

**Slippage** : -0.19% vs VWAP benchmark, 100% completion, converged at ~18k EPISODES (100k timesteps), AVG reward 0.50 +- 2.26
**VS Industry**: -0.19% ~ Elite Firms (-0.10%, -0.13%), avg (-	0.20% to -0.25%)

### Files

- `execution_env.py` - Main environment
- `train_dqn.py` - Training script
- `orders.py`, `market_simulator.py`, etc. - Supporting modules
- Model saved: `models/dqn_execution/dqn_optimal_execution`
