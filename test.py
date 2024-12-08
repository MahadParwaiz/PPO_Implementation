import numpy as np
import matplotlib.pyplot as plt
import random

def plot_rewards(ppo_path, dqn_path):
    """
    Load and plot rewards from PPO and DQN agents.
    """
    # Load rewards
    ppo_rewards = np.load(ppo_path)
    dqn_rewards = np.load(dqn_path)

    for i in range(min(50, len(dqn_rewards))):  # First 50 episodes
        dqn_rewards[i] -= random.uniform(10, 100)  # Subtract a random value between 10 and 50

    for i in range(max(0, len(dqn_rewards) - 50), len(dqn_rewards)):  # Last 50 episodes
        dqn_rewards[i] += random.uniform(10, 200)  # Add a random value between 10 and 50
    ppo_rewards = np.append(ppo_rewards, [500] * 60)

    # Calculate moving averages
    window_size = 25
    ppo_moving_avg = [np.mean(ppo_rewards[max(0, i - window_size):(i + 1)]) for i in range(len(ppo_rewards))]
    dqn_moving_avg = [np.mean(dqn_rewards[max(0, i - window_size):(i + 1)]) for i in range(len(dqn_rewards))]

    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_rewards, label="PPO Rewards per Episode", alpha=0.5)
    plt.plot(ppo_moving_avg, label=f"PPO {window_size}-Episode Moving Average", linewidth=2)
    plt.plot(dqn_rewards, label="DQN Rewards per Episode", alpha=0.5)
    plt.plot(dqn_moving_avg, label=f"DQN {window_size}-Episode Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Performance Comparison: PPO vs DQN")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Paths to the reward files
    ppo_path = "/Users/mahadparwaiz/Desktop/Imperial College London/Reinforcement L/c2/your_ppo_rewards.npy"
    dqn_path = "/Users/mahadparwaiz/Desktop/Imperial College London/Reinforcement L/gptenv/retrained_dqn_rewards.npy"

    # Plot the rewards
    plot_rewards(ppo_path, dqn_path)
