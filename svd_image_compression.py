import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Try to download image with proper headers
def download_image():
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/JPEG_example_flower.jpg/640px-JPEG_example_flower.jpg"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating a synthetic test image instead...\n")
        return create_synthetic_image()

def create_synthetic_image():
    """Create a simple test image with patterns"""
    size = 200
    img = np.zeros((size, size))
    
    # Add some circles and patterns
    center = size // 2
    y, x = np.ogrid[:size, :size]
    
    # Circle 1
    mask1 = (x - center)**2 + (y - center)**2 <= 40**2
    img[mask1] = 200
    
    # Circle 2
    mask2 = (x - 60)**2 + (y - 60)**2 <= 25**2
    img[mask2] = 150
    
    # Add some gradients
    img += np.linspace(0, 50, size).reshape(1, -1)
    
    return img.astype(np.uint8)

# Get the image
img_array = download_image()

print(f"Image shape: {img_array.shape}")
print(f"Image values range: {img_array.min()} to {img_array.max()}")

# Display the original image
plt.figure(figsize=(8, 6))
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()
# Perform SVD on the image
U, singular_values, VT = np.linalg.svd(img_array, full_matrices=False)

print(f"\nSVD Decomposition:")
print(f"U shape: {U.shape}")
print(f"Singular values shape: {singular_values.shape}")
print(f"V^T shape: {VT.shape}")

print(f"\nFirst 10 singular values:")
print(singular_values[:10])

# Plot singular values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(singular_values, 'b-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('All Singular Values')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(singular_values[:50], 'r-', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('First 50 Singular Values')
plt.grid(True)

plt.tight_layout()
plt.show()

# Image Compression: Reconstruct using only top k singular values
def compress_image(U, singular_values, VT, k):
    """Reconstruct image using only top k singular values"""
    # Create diagonal matrix with only top k values
    Sigma_k = np.zeros((k, k))
    np.fill_diagonal(Sigma_k, singular_values[:k])
    
    # Truncated matrices
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    
    # Reconstruct
    compressed = U_k @ Sigma_k @ VT_k
    
    return compressed

total = np.sum(singular_values)
sing = np.cumsum(singular_values) / total
k_90 = np.argmax(sing >= 0.90)

# Try different compression levels
k_values = [10, 20, 50, 100, k_90]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original image
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Compressed versions
for idx, k in enumerate(k_values, start=1):
    compressed = compress_image(U, singular_values, VT, k)
    axes[idx].imshow(compressed, cmap='gray')
    
    # Calculate compression ratio
    original_size = img_array.shape[0] * img_array.shape[1]
    compressed_size = k * (img_array.shape[0] + img_array.shape[1] + 1)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    axes[idx].set_title(f'k={k} ({compression_ratio:.1f}% smaller)')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Print compression details
print("\n" + "="*60)
print("COMPRESSION ANALYSIS")
print("="*60)
original_size = img_array.shape[0] * img_array.shape[1]
print(f"Original image: {img_array.shape[0]}×{img_array.shape[1]} = {original_size:,} values")
print()

for k in k_values:
    # We need to store: U_k (200×k) + singular_values (k) + VT_k (k×200)
    compressed_size = k * (img_array.shape[0] + img_array.shape[1] + 1)
    compression_ratio = (1 - compressed_size / original_size) * 100
    print(f"k={k:3d}: Store {compressed_size:,} values ({compression_ratio:.1f}% compression)")

