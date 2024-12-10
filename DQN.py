model_path = "/Users/mahadparwaiz/Desktop/Imperial College London/Reinforcement L/gptenv/rl-baselines3-zoo/rl-trained-agents/CartPole-v1.zip"

from stable_baselines3 import DQN
import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make("CartPole-v1")

# Train a new DQN model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save the model
model.save("retrained_dqn_cartpole")

# Evaluate the model
num_episodes = 200
total_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Save the rewards
np.save("retrained_dqn_rewards.npy", total_rewards)
print("Retrained DQN rewards saved to 'retrained_dqn_rewards.npy'.")
