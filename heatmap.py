import numpy as np
import matplotlib.pyplot as plt

# Discretize the state space for CartPole-v1 (simplified example)
position_bins = np.linspace(-2.4, 2.4, 50)  # Cart position range
angle_bins = np.linspace(-0.209, 0.209, 50)  # Pole angle range

# Simulate a value function for demonstration purposes
# Here, values increase as the pole angle nears zero and the cart position nears zero
value_function = np.zeros((len(position_bins), len(angle_bins)))

for i, pos in enumerate(position_bins):
    for j, angle in enumerate(angle_bins):
        value_function[i, j] = np.exp(-((pos / 2.4) ** 2 + (angle / 0.209) ** 2))

# Generate a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(value_function, extent=[-0.209, 0.209, -2.4, 2.4], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Value Function')
plt.title('Heatmap of Simulated Value Function for CartPole-v1')
plt.xlabel('Pole Angle (radians)')
plt.ylabel('Cart Position (meters)')
plt.show()