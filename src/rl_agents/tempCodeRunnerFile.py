
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
    total_timesteps = 50000                     # Total training timesteps (can be adjusted)
    
    model.learn(
        total_timesteps=total_timesteps,        # Total training timesteps
        callback=callback,                      # Custom callback for logging
        log_interval=100,                       # Log every 100 steps 
        progress_bar=True                       # Progress bar display
    )
    print("✓ Training complete")
    
    
    # 4. SAVE TRAINED MODEL
    print("\n[4] Saving trained model...")
    os.makedirs("models/dqn_execution", exist_ok=True)          # Create directory if it doesn't exist
    model_path= "models/dqn_execution/dqn_optimal_execution"    
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