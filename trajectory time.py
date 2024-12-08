import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# Disable eager execution for compatibility with TensorFlow 1.x-like behavior
tf.compat.v1.disable_eager_execution()

class ValueNetwork:
    def __init__(self, num_features, hidden_size=64, learning_rate=1e-3):
        self.num_features = num_features
        self.hidden_size = hidden_size
        # Separate graph
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()
            with tf.compat.v1.variable_scope("value_network"):
                self.observations = tf.compat.v1.placeholder(shape=[None, self.num_features], dtype=tf.float32)

                # Two-layer MLP
                layer_1 = tf.keras.layers.Dense(units=self.hidden_size, activation=tf.nn.relu)(self.observations)
                layer_2 = tf.keras.layers.Dense(units=self.hidden_size, activation=tf.nn.relu)(layer_1)
                self.output = tf.keras.layers.Dense(units=1, activation=None)(layer_2)
                self.output = tf.reshape(self.output, [-1])

                # Target values
                self.rollout = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
                self.loss = tf.reduce_mean((self.rollout - self.output) ** 2)
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)
                
                init = tf.compat.v1.global_variables_initializer()
                self.session.run(init)

    def get(self, states):
        return self.session.run(self.output, feed_dict={self.observations: states})

    def update(self, states, targets):
        self.session.run(self.train_op, feed_dict={self.observations: states, self.rollout: targets})

class PPOPolicyNetwork:
    def __init__(self, num_features, num_actions, hidden_size=64, epsilon=0.2, learning_rate=1e-3, entropy_coef=0.01):
        self.num_features = num_features
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()
            with tf.compat.v1.variable_scope("policy_network"):
                self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)
                self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
                self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
                self.old_probabilities = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

                # Two-layer MLP for policy
                layer_1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu)(self.observations)
                layer_2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu)(layer_1)
                logits = tf.keras.layers.Dense(units=num_actions, activation=None)(layer_2)
                self.output = tf.nn.softmax(logits)

                # New probabilities of chosen actions
                self.new_probs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
                self.ratio = self.new_probs / (self.old_probabilities + 1e-10)

                # PPO Objectives
                surr1 = self.ratio * self.advantages
                surr2 = tf.clip_by_value(self.ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Entropy bonus
                entropy = -tf.reduce_mean(tf.reduce_sum(self.output * tf.math.log(self.output + 1e-10), axis=1))
                self.loss = policy_loss - self.entropy_coef * entropy

                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)

                init = tf.compat.v1.global_variables_initializer()
                self.session.run(init)

    def get_dist(self, states):
        return self.session.run(self.output, feed_dict={self.observations: states})

    def update(self, states, chosen_actions, old_probabilities, advantages, epochs=10, batch_size=64):
        # Perform multiple epochs of minibatch updates
        data_size = len(states)
        indices = np.arange(data_size)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                feed = {
                    self.observations: states[batch_idx],
                    self.chosen_actions: chosen_actions[batch_idx],
                    self.old_probabilities: old_probabilities[batch_idx],
                    self.advantages: advantages[batch_idx]
                }
                self.session.run(self.train_op, feed_dict=feed)

# PPO class and main execution remain unchanged.



class PPO:
    def __init__(
        self,
        env,
        num_features=4,
        num_actions=2,
        gamma=0.99,
        lam=0.95,
        epsilon=0.2,
        value_learning_rate=1e-3,
        policy_learning_rate=1e-3,
        hidden_size=64,
        entropy_coef=0.01,
        steps_per_batch=2048
    ):
        self.env = env
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.steps_per_batch = steps_per_batch

        # Normalization buffers
        self.state_mean = np.zeros(num_features)
        self.state_var = np.ones(num_features)

        self.Pi = PPOPolicyNetwork(
            num_features=num_features,
            num_actions=num_actions,
            hidden_size=hidden_size,
            epsilon=epsilon,
            learning_rate=policy_learning_rate,
            entropy_coef=entropy_coef
        )
        self.V = ValueNetwork(num_features, hidden_size=hidden_size, learning_rate=value_learning_rate)
        self.all_episode_rewards = []
        self.episode = 0

    def normalize_state(self, state):
        return (state - self.state_mean) / (np.sqrt(self.state_var) + 1e-8)

    def update_running_stats(self, states):
        # Update mean and variance for state normalization
        old_mean = self.state_mean.copy()
        old_var = self.state_var.copy()
        new_mean = np.mean(states, axis=0)
        new_var = np.var(states, axis=0)

        # Simple moving average
        alpha = 0.99
        self.state_mean = alpha * old_mean + (1 - alpha) * new_mean
        self.state_var = alpha * old_var + (1 - alpha) * new_var

    def discount_rewards(self, rewards):
        # Compute discounted rewards
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = running_total * self.gamma + rewards[t]
            discounted[t] = running_total
        return discounted

    def calculate_advantages(self, rewards, values, next_value):
        # GAE-Lambda advantages
        deltas = rewards + self.gamma * np.append(values[1:], [next_value]) - values
        advantages = np.zeros_like(deltas, dtype=np.float32)
        running_adv = 0
        for t in reversed(range(len(deltas))):
            running_adv = deltas[t] + self.gamma * self.lam * running_adv
            advantages[t] = running_adv
        return advantages

    def collect_batch(self):
        # Collect a batch of experiences
        states = []
        actions = []
        rewards = []
        old_probs = []
        values = []

        steps = 0
        while steps < self.steps_per_batch:
            obs, _ = self.env.reset()
            done = False
            ep_rewards = 0
            while not done and steps < self.steps_per_batch:
                # Normalize state
                s = self.normalize_state(np.array(obs).reshape(1, -1))
                action_probs = self.Pi.get_dist(s)[0]

                action = np.random.choice(range(self.num_actions), p=action_probs)
                a_binarized = np.zeros(self.num_actions)
                a_binarized[action] = 1

                v = self.V.get(s)[0]

                # Step
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(obs)
                actions.append(a_binarized)
                rewards.append(reward)
                old_probs.append(action_probs[action])
                values.append(v)

                obs = next_obs
                ep_rewards += reward
                steps += 1

            self.all_episode_rewards.append(ep_rewards)
            self.episode += 1

        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)

        # Update running stats for normalization
        self.update_running_stats(states)
        # Re-normalize states after updating stats
        states = self.normalize_state(states)

        # Next value for advantage calculation
        # Since we ended a batch arbitrarily, take last state:
        # If we consider them as episodic tasks, next_value = 0 when done.
        next_value = 0.0  
        advantages = self.calculate_advantages(rewards, values, next_value)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return states, actions, old_probs, returns, advantages

    def train(self, total_timesteps=100000, ppo_epochs=10, batch_size=64):
        while True:
            states, actions, old_probs, returns, advantages = self.collect_batch()

            # Update Value Network
            self.V.update(states, returns)

            # Update Policy Network using multiple epochs
            self.Pi.update(states, actions, old_probs, advantages, epochs=ppo_epochs, batch_size=batch_size)

            if len(self.all_episode_rewards) > 100:
                avg_reward = np.mean(self.all_episode_rewards[-100:])
                print(f"Episode: {self.episode}, Avg Reward (Last 100 eps): {avg_reward}")
                if avg_reward >= 195:
                    print("Environment solved!")
                    break
            if self.episode > 1000:  # Safety break if taking too long
                print("Reached 1000 episodes without solving, stopping.")
                break
# Use the above modifications to resolve your issue.

# Main execution
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = PPO(env, num_features=4, num_actions=2, gamma=0.99, lam=1,
                 epsilon=0.2, value_learning_rate=1e-4, policy_learning_rate=0.01,
                 hidden_size=100, entropy_coef=0.01, steps_per_batch=2048)
    agent.train(total_timesteps=100000)

    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(agent.all_episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

    # Moving average of Total Rewards
    window = 25
    rewards = agent.all_episode_rewards
    moving_avg_rewards = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward')
    plt.plot(moving_avg_rewards, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress with Moving Average')
    plt.legend()
    plt.show()
