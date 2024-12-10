import matplotlib.pyplot as plt 
import numpy as np 
ratios = np.linspace(0, 2, 200)  # Probability ratios from 0 to 2
advantages = [1, -1]  # Example advantages (positive and negative)  
epsilon=0.2

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