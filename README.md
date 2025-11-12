# Hierarchical RL Execution (Simulation)

**Hierarchical reinforcement learning framework for optimal intraday trade execution using simulated market environments.**

## Overview

- **Strategic Layer:** PPO agent selects execution pace
- **Tactical Layer:** DQN agent manages order slicing based on pace guidance
- **Baselines:** VWAP, TWAP, single-agent RL (for controlled benchmarking)

All training and evaluation conducted on **synthetic price data** to enable reproducible research and robust algorithm analysis.  
Use this repo for developing and benchmarking methodologies before validating on real market data.

## Repository Structure

```
/src/hierarchical/        # RL architecture and environments (simulation)
/models/                  # Trained models (simulation)
/analysis/                # Results and analysis scripts
```

## How to Run

1. **Install requirements:**  
   `pip install -r requirements.txt`
2. **Train agents:**  
   See scripts in `/src/hierarchical/`
3. **Evaluate against baselines:**  
   Run analysis scripts in `/analysis/`

## Note

For **real market data validation and out-of-sample backtesting**, refer to the companion repository 

***

Author: @ssrhaso
