import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo

# Disable eager execution for compatibility with TensorFlow 1.x code
tf.compat.v1.disable_eager_execution()

class ValueNetwork:
    def __init__(self, num_features, hidden_size, learning_rate=0.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        # Create a separate graph for the value network
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            # Start a TensorFlow session
            self.session = tf.compat.v1.Session()
            # Use a variable scope to prevent naming conflicts
            with tf.compat.v1.variable_scope("value_network"):
                # Placeholder for input observations
                self.observations = tf.compat.v1.placeholder(
                    shape=[None, self.num_features], dtype=tf.float32
                )
                # Define network weights
                self.W1 = tf.compat.v1.get_variable(
                    "W1", shape=[self.num_features, self.hidden_size],
                    initializer=tf.compat.v1.keras.initializers.GlorotUniform()
                )
                self.W2 = tf.compat.v1.get_variable(
                    "W2", shape=[self.hidden_size, 1],
                    initializer=tf.compat.v1.keras.initializers.GlorotUniform()
                )
                # Forward pass
                self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
                self.output = tf.reshape(tf.matmul(self.layer_1, self.W2), [-1])
                # Placeholder for target values
                self.rollout = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
                # Define loss function (mean squared error)
                self.loss = tf.compat.v1.losses.mean_squared_error(
                    labels=self.rollout, predictions=self.output
                )
                # Optimizer
                self.optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                )
                self.train_op = self.optimizer.minimize(self.loss)
                # Initialize variables
                init = tf.compat.v1.global_variables_initializer()
                self.session.run(init)

    def get(self, states):
        # Predict value for given states
        value = self.session.run(
            self.output, feed_dict={self.observations: states}
        )
        return value

    def update(self, states, discounted_rewards):
        # Update network weights based on the loss
        self.session.run(
            self.train_op,
            feed_dict={
                self.observations: states,
                self.rollout: discounted_rewards,
            },
        )

class PPOPolicyNetwork:
    def __init__(
        self,
        num_features,
        layer_1_size,
        layer_2_size,
        layer_3_size,
        num_actions,
        epsilon=0.2,
        learning_rate=9e-4,
    ):
        # Create a separate graph for the policy network
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()
            # Use a variable scope to prevent naming conflicts
            with tf.compat.v1.variable_scope("policy_network"):
                # Placeholder for input observations
                self.observations = tf.compat.v1.placeholder(
                    shape=[None, num_features], dtype=tf.float32
                )
                # Define network weights with variance scaling initializer
                self.W1 = tf.compat.v1.get_variable(
                    "W1",
                    shape=[num_features, layer_1_size],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(),
                )
                self.W2 = tf.compat.v1.get_variable(
                    "W2",
                    shape=[layer_1_size, layer_2_size],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(),
                )
                self.W3 = tf.compat.v1.get_variable(
                    "W3",
                    shape=[layer_2_size, layer_3_size],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(),
                )
                self.W4 = tf.compat.v1.get_variable(
                    "W4",
                    shape=[layer_3_size, num_actions],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(),
                )
                # Forward pass
                self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
                self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W2))
                self.layer_3 = tf.nn.relu(tf.matmul(self.layer_2, self.W3))
                logits = tf.matmul(self.layer_3, self.W4)
                self.output = tf.nn.softmax(logits)
                # Placeholders for training
                self.advantages = tf.compat.v1.placeholder(
                    shape=[None], dtype=tf.float32
                )
                self.chosen_actions = tf.compat.v1.placeholder(
                    shape=[None, num_actions], dtype=tf.float32
                )
                self.old_probabilities = tf.compat.v1.placeholder(
                    shape=[None], dtype=tf.float32
                )
                # Calculate probabilities for chosen actions
                self.new_probs = tf.reduce_sum(
                    self.chosen_actions * self.output, axis=1
                )
                # Calculate ratio for PPO
                self.ratio = self.new_probs / (self.old_probabilities + 1e-10)
                # Surrogate losses
                self.surr1 = self.ratio * self.advantages
                self.surr2 = (
                    tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon)
                    * self.advantages
                )
                # Loss function
                self.loss = -tf.reduce_mean(tf.minimum(self.surr1, self.surr2))
                # Optimizer
                self.optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                )
                self.train_op = self.optimizer.minimize(self.loss)
                # Initialize variables
                init = tf.compat.v1.global_variables_initializer()
                self.session.run(init)

    def get_dist(self, states):
        # Get action probability distribution for given states
        dist = self.session.run(
            self.output, feed_dict={self.observations: states}
        )
        return dist

    def update(self, states, chosen_actions, old_probabilities, advantages):
        # Update network weights based on the loss
        self.session.run(
            self.train_op,
            feed_dict={
                self.observations: states,
                self.chosen_actions: chosen_actions,
                self.old_probabilities: old_probabilities,
                self.advantages: advantages,
            },
        )

class PPO:
    def __init__(
        self,
        env,
        num_features=1,
        num_actions=1,
        gamma=0.98,
        lam=1,
        epsilon=0.2,
        value_network_lr=0.1,
        policy_network_lr=9e-4,
        value_network_hidden_size=100,
        policy_network_hidden_size_1=10,
        policy_network_hidden_size_2=10,
        policy_network_hidden_size_3=10,
    ):
        self.env = env
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        # Initialize policy and value networks
        self.Pi = PPOPolicyNetwork(
            num_features=num_features,
            num_actions=num_actions,
            layer_1_size=policy_network_hidden_size_1,
            layer_2_size=policy_network_hidden_size_2,
            layer_3_size=policy_network_hidden_size_3,
            epsilon=epsilon,
            learning_rate=policy_network_lr,
        )
        self.V = ValueNetwork(
            num_features, value_network_hidden_size, learning_rate=value_network_lr
        )
        self.all_episode_rewards = []  # Store rewards for all episodes

    def discount_rewards(self, rewards):
        # Compute discounted rewards
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = running_total * self.gamma + rewards[t]
            discounted[t] = running_total
        return discounted

    def calculate_advantages(self, rewards, values):
        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        deltas = np.zeros_like(rewards, dtype=np.float32)
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            deltas[t] = delta
            next_value = values[t]
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.lam * advantage
            advantages[t] = advantage
        return advantages

    def run_model(self):
        episode = 1
        running_reward = []
        render = False  # Set to False to speed up training
        while True:
            # Reset environment
            observation, info = self.env.reset()
            is_terminal = False
            ep_rewards = []
            ep_actions = []
            ep_states = []
            action_probs_list = []  # To store action probabilities
            score = 0
            while not is_terminal:
                if render:
                    self.env.render()
                s0 = np.array(observation)
                # Get action probabilities
                action_probs = self.Pi.get_dist(s0[np.newaxis, :])[0]
                action_probs_list.append(action_probs)  # Store for plotting
                # Sample action based on probabilities
                action = np.random.choice(range(self.num_actions), p=action_probs)
                # One-hot encode the action
                a_binarized = np.zeros(self.num_actions)
                a_binarized[action] = 1
                # Step the environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                is_terminal = terminated or truncated

                score += reward
                # Store experiences
                ep_actions.append(a_binarized)
                ep_rewards.append(reward)
                ep_states.append(s0)
                observation = next_observation
                if is_terminal:
                    # Prepare data for training
                    ep_actions = np.vstack(ep_actions)
                    ep_rewards = np.array(ep_rewards, dtype=np.float32)
                    ep_states = np.vstack(ep_states)
                    # Update Value Network
                    values = self.V.get(ep_states)
                    targets = self.discount_rewards(ep_rewards)
                    self.V.update(ep_states, targets)
                    # Calculate advantages
                    advantages = self.calculate_advantages(ep_rewards, values)
                    # Normalize advantages
                    advantages = (advantages - np.mean(advantages)) / (
                        np.std(advantages) + 1e-10
                    )
                    # Update Policy Network
                    old_probs = self.Pi.get_dist(ep_states)
                    old_selected_probs = np.sum(ep_actions * old_probs, axis=1)
                    self.Pi.update(ep_states, ep_actions, old_selected_probs, advantages)
                    running_reward.append(score)
                    self.all_episode_rewards.append(score)  # Store the score
                    avg_score = np.mean(running_reward[-25:])
                    if episode % 25 == 0:
                        avg_score = np.mean(running_reward[-25:])
                        print(f"Episode: {episode} Average Score: {avg_score}")
                    if avg_score >= 500:
                        print(f"Episode {episode}: Reached score of {score}, stopping training.")
                        return
                    episode += 1


# Main execution
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import gym
    from gym.wrappers import RecordVideo

    # Ensure 'CartPole-v1' supports 'rgb_array' render mode
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env._max_episode_steps = 500  # or a smaller number if desired

    # Wrap the environment with RecordVideo
    #env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: e % 50 == 0)
    # Reset the environment before starting
    env.reset()

    agent = PPO(
        env,
        num_features=4,
        num_actions=2,
        gamma=0.98,
        lam=1,
        epsilon=0.2,
        value_network_lr=0.001,
        policy_network_lr=0.01,
        value_network_hidden_size=100,
        policy_network_hidden_size_1=40,
        policy_network_hidden_size_2=35,
        policy_network_hidden_size_3=30,
    )
    agent.run_model()


    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(agent.all_episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()









    # Plot the moving average of Total Rewards
    window = 25
    rewards = agent.all_episode_rewards
    moving_avg_rewards = [np.mean(rewards[max(0, i-window):(i+1)]) for i in range(len(rewards))]
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.plot(moving_avg_rewards, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress with Moving Average')
    plt.legend()
    plt.show()

    epsilon = agent.epsilon  # Get the epsilon value used in PPO
    ratios = np.linspace(0, 2, 200)  # Probability ratios from 0 to 2

    advantages = [1, -1]  # Example advantages (positive and negative)

    plt.figure(figsize=(10, 6))

    for adv in advantages:
        surr1 = ratios * adv
        surr2 = np.clip(ratios, 1 - epsilon, 1 + epsilon) * adv
        clipped_surrogate = np.minimum(surr1, surr2)
        label = f'Advantage {adv}'
        plt.plot(ratios, clipped_surrogate, label=label)

    # Plot vertical lines indicating the clipping boundaries
    plt.axvline(1 - epsilon, color='grey', linestyle='--', label='Clipping Boundaries')
    plt.axvline(1 + epsilon, color='grey', linestyle='--')

    plt.xlabel('Probability Ratio')
    plt.ylabel('Clipped Surrogate Loss')
    plt.title('Visualization of PPO Clipping Function')
    plt.legend()
    plt.grid(True)
    plt.show()



    # Define the range for cart position and pole angle
    cart_pos_min, cart_pos_max = -4.8, 4.8
    pole_angle_min, pole_angle_max = -0.418, 0.418  # ~24 degrees in radians

    # Create a grid of cart positions and pole angles
    num_points = 100
    cart_positions = np.linspace(cart_pos_min, cart_pos_max, num_points)
    pole_angles = np.linspace(pole_angle_min, pole_angle_max, num_points)
    cart_positions_grid, pole_angles_grid = np.meshgrid(cart_positions, pole_angles)

    # Initialize arrays to store values and action probabilities
    value_function_grid = np.zeros_like(cart_positions_grid)
    action_probability_grid = np.zeros_like(cart_positions_grid)

    # Fixed variables
    cart_velocity = 0.0
    pole_angular_velocity = 0.0

    # Loop over the grid and compute value function and action probabilities
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([
                cart_positions_grid[i, j],
                cart_velocity,
                pole_angles_grid[i, j],
                pole_angular_velocity
            ]).reshape(1, -1)
            # Compute value function
            value = agent.V.get(state)[0]
            value_function_grid[i, j] = value
            # Compute action probabilities
            action_probs = agent.Pi.get_dist(state)[0]
            action_probability_grid[i, j] = action_probs[0]  # Probability of action 0 (left)

    # Plot the value function heatmap
    plt.figure(figsize=(12, 6))
    plt.title('Value Function Heatmap')
    plt.xlabel('Cart Position')
    plt.ylabel('Pole Angle')
    plt.imshow(value_function_grid, extent=[cart_pos_min, cart_pos_max, pole_angle_min, pole_angle_max], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Value Function')
    plt.show()

    # Plot the action probability heatmap
    plt.figure(figsize=(12, 6))
    plt.title('Action Probability Heatmap (Action 0 - Left)')
    plt.xlabel('Cart Position')
    plt.ylabel('Pole Angle')
    plt.imshow(action_probability_grid, extent=[cart_pos_min, cart_pos_max, pole_angle_min, pole_angle_max], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability of Action 0')
    plt.show()
