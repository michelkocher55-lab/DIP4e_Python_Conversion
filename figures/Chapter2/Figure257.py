
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = [10, 10]

# Covariance matrices
C1 = [[5, 0], [0, 5]]
C2 = [[12, 0], [0, 5]]
C3 = [[12, 6], [6, 5]]
C4 = [[12, -6], [-6, 5]]

# Generate random samples
# mvnrnd(MU, SIGMA, cases)
# numpy: multivariate_normal(mean, cov, size)

n_samples = 1000

r1 = np.random.multivariate_normal(m, C1, n_samples)
r2 = np.random.multivariate_normal(m, C2, n_samples)
r3 = np.random.multivariate_normal(m, C3, n_samples)
r4 = np.random.multivariate_normal(m, C4, n_samples)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

limit = [0, 20, 0, 20]

# Subplot 1
axes[0].plot(r1[:, 0], r1[:, 1], '.')
axes[0].axis(limit)
axes[0].set_aspect('equal', adjustable='box') # axis square
axes[0].set_title('Covariance: [5 0; 0 5]')

# Subplot 2
axes[1].plot(r2[:, 0], r2[:, 1], '.')
axes[1].axis(limit)
axes[1].set_aspect('equal', adjustable='box')
axes[1].set_title('Covariance: [12 0; 0 5]')

# Subplot 3
axes[2].plot(r3[:, 0], r3[:, 1], '.')
axes[2].axis(limit)
axes[2].set_aspect('equal', adjustable='box')
axes[2].set_title('Covariance: [12 6; 6 5]')

# Subplot 4
axes[3].plot(r4[:, 0], r4[:, 1], '.')
axes[3].axis(limit)
axes[3].set_aspect('equal', adjustable='box')
axes[3].set_title('Covariance: [12 -6; -6 5]')

plt.tight_layout()
plt.savefig('Figure257.png')
print("Saved Figure257.png")

plt.show()