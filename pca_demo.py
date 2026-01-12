import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic 2D data with correlation
np.random.seed(42)
n_samples = 300

# Create correlated data (elongated along a diagonal)
mean = [2, 3]
cov = [[3, 2.5],   # Covariance matrix
       [2.5, 3]]
data = np.random.multivariate_normal(mean, cov, n_samples)

print("Original data shape:", data.shape)
print("Data statistics:")
print(f"  X: mean={data[:, 0].mean():.2f}, std={data[:, 0].std():.2f}")
print(f"  Y: mean={data[:, 1].mean():.2f}, std={data[:, 1].std():.2f}")

# Visualize original data
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original 2D Data')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# STEP 1: Center the data (subtract mean)
data_centered = data - data.mean(axis=0)

print("\nCentered data statistics:")
print(f"  X: mean={data_centered[:, 0].mean():.6f}, std={data_centered[:, 0].std():.2f}")
print(f"  Y: mean={data_centered[:, 1].mean():.6f}, std={data_centered[:, 1].std():.2f}")

# STEP 2: Compute SVD
U, singular_values, VT = np.linalg.svd(data_centered.T, full_matrices=False)

# Principal components are the columns of U (or rows of VT transposed)
principal_components = U

print("\nSingular values:", singular_values)
print("Explained variance ratio:", singular_values**2 / np.sum(singular_values**2))

# STEP 3: Visualize principal components
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30, label='Data')

# Draw principal component arrows from the mean
mean_point = data.mean(axis=0)
scale = 3  # Scale for visualization

# First principal component (most variance)
pc1 = principal_components[:, 0] * scale
plt.arrow(mean_point[0], mean_point[1], pc1[0], pc1[1], 
          head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=3,
          label=f'PC1 (explains {(singular_values[0]**2 / np.sum(singular_values**2))*100:.1f}%)')

# Second principal component (least variance)
pc2 = principal_components[:, 1] * scale
plt.arrow(mean_point[0], mean_point[1], pc2[0], pc2[1],
          head_width=0.2, head_length=0.3, fc='blue', ec='blue', linewidth=3,
          label=f'PC2 (explains {(singular_values[1]**2 / np.sum(singular_values**2))*100:.1f}%)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA: Principal Components Overlaid on Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# STEP 4: Project data onto principal components
# Project onto PC1 only (dimensionality reduction: 2D -> 1D)
data_1d = data_centered @ principal_components[:, 0:1]

print(f"\nProjected to 1D shape: {data_1d.shape}")
print(f"We went from 2D to 1D, keeping 91% of variance!")

# Reconstruct back to 2D to visualize what we kept
data_reconstructed_1d = data_1d @ principal_components[:, 0:1].T + data.mean(axis=0)

# Also project onto both components (no reduction)
data_2d = data_centered @ principal_components
data_reconstructed_2d = data_2d @ principal_components.T + data.mean(axis=0)

# Visualize the projection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=30, c='blue')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Original Data (2D)')
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# Projected onto PC1 only (1D)
axes[1].scatter(data_reconstructed_1d[:, 0], data_reconstructed_1d[:, 1], 
                alpha=0.5, s=30, c='red')
axes[1].arrow(mean_point[0], mean_point[1], pc1[0], pc1[1], 
              head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=3, alpha=0.3)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('Projected onto PC1 (1D) - 91% variance kept')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

# Show the 1D representation directly
axes[2].scatter(data_1d, np.zeros_like(data_1d), alpha=0.5, s=30, c='green')
axes[2].set_xlabel('PC1 coordinate')
axes[2].set_ylabel('(collapsed)')
axes[2].set_title('Pure 1D Representation')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.5, 0.5)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("="*60)
print(f"Original: {data.shape[1]} dimensions")
print(f"Reduced:  1 dimension")
print(f"Information kept: 91%")
print(f"Storage reduction: {(1 - 1/data.shape[1])*100:.0f}%")
