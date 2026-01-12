"""
Neural Network from Scratch
Building a 2-layer neural network to understand Transformers fundamentals

Architecture:
    Input (784) → Hidden Layer (128, ReLU) → Output (10, Softmax)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize the neural network with random weights and zero biases.
        
        Args:
            input_size: Number of input features (784 for MNIST 28x28 images)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes (10 for digits 0-9)
        """
        np.random.seed(42)
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def relu(self, z):
        """ReLU activation: f(x) = max(0, x)"""
        return np.maximum(0, z)
    
    def softmax(self, z):
        """
        TODO: Implement the softmax activation function
        
        Softmax formula: softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
        
        This converts raw scores (logits) into probabilities that sum to 1.
        
        Args:
            z: Input array of shape (batch_size, num_classes)
        
        Returns:
            Array of same shape with probabilities (sum to 1 along axis 1)
        
        Hints:
            - Use np.exp(z) to compute e^z
            - Use np.sum(..., axis=1, keepdims=True) to sum along the class dimension
            - Divide to normalize
            
        Example:
            Input:  [[2.0, 1.0, 0.1]]
            Output: [[0.659, 0.242, 0.099]]  (sums to 1.0)
        """
        # YOUR CODE HERE
        # DIAGNOSTIC: Check for extreme values
        if np.any(np.abs(z) > 100):
            print(f"WARNING: Extreme z2 values detected!")
            print(f"  Min z2: {z.min():.2f}")
            print(f"  Max z2: {z.max():.2f}")
        
        exp_z = np.exp(z)
        result = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return result
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, 784)
        
        Returns:
            predictions: Output probabilities of shape (batch_size, 10)
        """
        # Layer 1: Linear + ReLU
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2: Linear + Softmax
        self.z2 = self.a1 @ self.W2 + self.b2
        self.predictions = self.softmax(self.z2)
        
        return self.predictions
    
    def predict(self, X):
        """Make predictions on input data"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def cross_entropy_loss(self, predictions, labels):
        size = len(predictions)
        prob = predictions[range(size), labels]
        entropy = -np.log(prob)
        return np.mean(entropy)
    
    def backward(self, X, labels, learning_rate):
        """
        Backward propagation - compute gradients and update weights.
        
        Args:
            X: Input data of shape (batch_size, 784)
            labels: True labels of shape (batch_size,)
            learning_rate: Step size for gradient descent
        """
        batch_size = len(X)
        
        # === STEP 1: Gradient at output layer (dL/dz2) ===
        # For softmax + cross-entropy, the gradient simplifies to:
        # dL/dz2 = predictions - one_hot(labels)
        
        # TODO: Create one-hot encoded labels
        # Hint: Initialize zeros array of shape (batch_size, 10)
        #       Then set the correct class positions to 1
        #       Use fancy indexing: one_hot[range(batch_size), labels] = 1
        one_hot = np.zeros((batch_size, 10))
        # YOUR CODE HERE to set the 1s at correct positions
        one_hot[range(batch_size), labels] = 1
        
        # TODO: Compute dL/dz2
        # dL_dz2 = ?
        
        predictions = self.predictions
        dL_dz2 = predictions - one_hot
        
        # === STEP 2: Gradients for output layer weights and biases ===
        
        # TODO: Compute dL/dW2
        dL_dW2 = self.a1.T @ dL_dz2
        
        # TODO: Compute dL/db2
        # Hint: Sum dL/dz2 across the batch dimension
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # === STEP 3: Gradient flowing back to hidden layer (dL/da1) ===
        
        # TODO: Compute dL/da1 using chain rule
        # We know: dL/dz2 and we need to go back through W2
        # Think: z2 = a1 @ W2 + b2, so how does a1 affect z2?
        dL_da1 = dL_dz2 @ self.W2.T
        
        # === STEP 4: Gradient through ReLU (dL/dz1) ===
        
        # TODO: Compute dL/dz1
        # ReLU derivative: 1 if z1 > 0, else 0
        # Hint: Create a mask where z1 > 0, then multiply element-wise
        relu_mask = (self.z1 > 0).astype(float)
        dL_dz1 = dL_da1 * relu_mask
        
        # === STEP 5: Gradients for hidden layer weights and biases ===
        
        # TODO: Compute dL/dW1
        dL_dW1 = X.T @ dL_dz1
        
        # TODO: Compute dL/db1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        # === STEP 6: Update weights using gradient descent ===
        
        # TODO: Update all parameters
        # Rule: parameter = parameter - learning_rate * gradient
        self.W2 = self.W2 - learning_rate * dL_dW2
        self.b2 = self.b2 - learning_rate * dL_db2
        self.W1 = self.W1 - learning_rate * dL_dW1
        self.b1 = self.b1 - learning_rate * dL_db1


def test_softmax():
    """Test your softmax implementation"""
    nn = NeuralNetwork()
    
    # Test case 1: Single sample
    z = np.array([[2.0, 1.0, 0.1]])
    result = nn.softmax(z)
    
    print("Test 1 - Single sample:")
    print(f"Input:  {z[0]}")
    print(f"Output: {result[0]}")
    print(f"Sum:    {np.sum(result[0]):.6f} (should be 1.0)")
    print()
    
    # Test case 2: Batch of samples
    z = np.array([
        [2.0, 1.0, 0.1],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ])
    result = nn.softmax(z)
    
    print("Test 2 - Batch of samples:")
    for i in range(len(z)):
        print(f"Sample {i+1} - Sum: {np.sum(result[i]):.6f} (should be 1.0)")


def test_loss():
    """Test cross-entropy loss implementation"""
    nn = NeuralNetwork()
    
    print("\n" + "=" * 60)
    print("Testing Cross-Entropy Loss")
    print("=" * 60)
    
    # Test case 1: Perfect predictions
    predictions = np.array([
        [0.0, 0.0, 1.0],  # Very confident it's class 2
        [1.0, 0.0, 0.0],  # Very confident it's class 0
    ])
    labels = np.array([2, 0])  # True labels match predictions
    
    loss = nn.cross_entropy_loss(predictions, labels)
    print(f"\nTest 1 - Perfect predictions:")
    print(f"Loss: {loss:.6f} (should be very close to 0)")
    
    # Test case 2: Terrible predictions
    predictions = np.array([
        [0.0, 0.0, 0.01],  # Says class 2 with only 1% confidence
        [0.01, 0.0, 0.0],  # Says class 0 with only 1% confidence
    ])
    labels = np.array([2, 0])  # True labels
    
    loss = nn.cross_entropy_loss(predictions, labels)
    print(f"\nTest 2 - Terrible predictions (1% confidence):")
    print(f"Loss: {loss:.6f} (should be large, around 4.6)")
    
    # Test case 3: Medium predictions
    predictions = np.array([
        [0.2, 0.3, 0.5],  # 50% confidence in class 2
        [0.6, 0.3, 0.1],  # 60% confidence in class 0
    ])
    labels = np.array([2, 0])
    
    loss = nn.cross_entropy_loss(predictions, labels)
    print(f"\nTest 3 - Medium predictions:")
    print(f"Loss: {loss:.6f} (should be moderate)")


def load_mnist_data():
    """
    Load MNIST dataset (handwritten digits 0-9)
    
    Returns:
        X_train: Training images (56000, 784)
        y_train: Training labels (56000,)
        X_test: Test images (14000, 784)
        y_test: Test labels (14000,)
    """
    print("Loading MNIST dataset...")
    
    # Load data from sklearn
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy().astype('float32')
    y = mnist.target.to_numpy().astype('int')
    
    # Normalize pixel values from [0, 255] to [0, 1]
    X = X / 255.0
    
    # Split into train/test (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    return X_train, y_train, X_test, y_test


def get_batches(X, y, batch_size):
    """
    Split data into mini-batches for training.
    
    Args:
        X: Input data (n_samples, 784)
        y: Labels (n_samples,)
        batch_size: Size of each batch
    
    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train(nn, X_train, y_train, X_test, y_test, 
          epochs=10, batch_size=32, learning_rate=0.1):
    """
    Train the neural network using mini-batch gradient descent.
    
    Args:
        nn: NeuralNetwork instance
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        epochs: Number of passes through the full dataset
        batch_size: Number of samples per gradient update
        learning_rate: Step size for gradient descent
    """
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    for epoch in range(epochs):
        # Training phase
        epoch_loss = 0
        n_batches = 0
        
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            # Forward pass
            predictions = nn.forward(X_batch)
            
            # Compute loss
            batch_loss = nn.cross_entropy_loss(predictions, y_batch)
            epoch_loss += batch_loss
            n_batches += 1
            
            # Backward pass (computes gradients and updates weights)
            nn.backward(X_batch, y_batch, learning_rate)
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / n_batches
        
        # Evaluation phase (every epoch)
        train_acc = evaluate(nn, X_train, y_train)
        test_acc = evaluate(nn, X_test, y_test)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f} - "
              f"Train Acc: {train_acc:.2%} - "
              f"Test Acc: {test_acc:.2%}")
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)


def evaluate(nn, X, y):
    """
    Evaluate the network's accuracy on a dataset.
    
    Args:
        nn: NeuralNetwork instance
        X: Input data
        y: True labels
    
    Returns:
        Accuracy (float between 0 and 1)
    """
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy


def run_experiment(X_train, y_train, X_test, y_test, 
                   hidden_size=128, epochs=10, batch_size=32, learning_rate=0.01,
                   experiment_name="Experiment"):
    """
    Run a single training experiment and return results.
    
    Returns:
        dict with 'final_test_acc', 'final_train_acc', 'final_loss', 'history'
    """
    print(f"\n{'='*60}")
    print(f"🧪 {experiment_name}")
    print(f"{'='*60}")
    print(f"Hidden size: {hidden_size}, Epochs: {epochs}, LR: {learning_rate}")
    print()
    
    # Create network with specified architecture
    nn = NeuralNetwork(hidden_size=hidden_size)
    
    # Track history
    history = {
        'train_acc': [],
        'test_acc': [],
        'loss': []
    }
    
    # Training loop (modified to track history)
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            predictions = nn.forward(X_batch)
            batch_loss = nn.cross_entropy_loss(predictions, y_batch)
            epoch_loss += batch_loss
            n_batches += 1
            nn.backward(X_batch, y_batch, learning_rate)
        
        avg_loss = epoch_loss / n_batches
        train_acc = evaluate(nn, X_train, y_train)
        test_acc = evaluate(nn, X_test, y_test)
        
        # Store history
        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f} - "
              f"Train Acc: {train_acc:.2%} - "
              f"Test Acc: {test_acc:.2%}")
    
    print(f"\n✅ Final Test Accuracy: {test_acc:.2%}")
    
    return {
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'final_loss': avg_loss,
        'history': history,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'epochs': epochs
    }


def compare_experiments(results_list):
    """
    Print a comparison table of experiment results.
    """
    print("\n" + "="*80)
    print("📊 EXPERIMENT COMPARISON")
    print("="*80)
    print(f"{'Experiment':<25} {'Hidden':<10} {'LR':<10} {'Epochs':<10} {'Final Acc':<12} {'Final Loss'}")
    print("-"*80)
    
    for i, result in enumerate(results_list, 1):
        name = f"Experiment {i}"
        hidden = result['hidden_size']
        lr = result['learning_rate']
        epochs = result['epochs']
        acc = f"{result['final_test_acc']:.2%}"
        loss = f"{result['final_loss']:.4f}"
        
        print(f"{name:<25} {hidden:<10} {lr:<10} {epochs:<10} {acc:<12} {loss}")
    
    print("="*80)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network from Scratch - Testing")
    print("=" * 60)
    print()
    
    # Test basic functions
    # test_softmax()
    # test_loss()
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER EXPERIMENTS")
    print("=" * 60)
    
    # Load data once
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Store all results for comparison
    all_results = []
    
    # ========================================
    # EXPERIMENT SET 1: Learning Rate
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT SET 1: Learning Rate Impact")
    print("# (Keep hidden_size=128, epochs=10 constant)")
    print("#"*60)
    
    # Experiment 1.1: Very small learning rate
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=10, learning_rate=0.001,
                           experiment_name="LR = 0.001 (Very Small)")
    all_results.append(result)
    
    # Experiment 1.2: Your working learning rate
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=10, learning_rate=0.01,
                           experiment_name="LR = 0.01 (Sweet Spot)")
    all_results.append(result)
    
    # Experiment 1.3: Larger learning rate (but not exploding)
    # result = run_experiment(X_train, y_train, X_test, y_test,
    #                        hidden_size=128, epochs=10, learning_rate=0.05,
    #                        experiment_name="LR = 0.05 (Large)")
    # all_results.append(result)
    
    # Compare Set 1
    print("\n📈 Learning Rate Comparison:")
    compare_experiments(all_results)
    
    # ========================================
    # EXPERIMENT SET 2: Hidden Layer Size
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT SET 2: Hidden Layer Size Impact")
    print("# (Keep lr=0.01, epochs=10 constant)")
    print("#"*60)
    
    set2_results = []
    
    # Experiment 2.1: Small network
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=64, epochs=10, learning_rate=0.01,
                           experiment_name="Hidden = 64 (Small)")
    set2_results.append(result)
    
    # Experiment 2.2: Medium network (baseline)
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=10, learning_rate=0.01,
                           experiment_name="Hidden = 128 (Medium)")
    set2_results.append(result)
    
    # Experiment 2.3: Large network
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=256, epochs=10, learning_rate=0.01,
                           experiment_name="Hidden = 256 (Large)")
    set2_results.append(result)
    
    # Compare Set 2
    print("\n📈 Hidden Size Comparison:")
    compare_experiments(set2_results)
    all_results.extend(set2_results)
    
    # ========================================
    # EXPERIMENT SET 3: Number of Epochs
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT SET 3: Training Duration Impact")
    print("# (Keep hidden_size=128, lr=0.01 constant)")
    print("#"*60)
    
    set3_results = []
    
    # Experiment 3.1: Short training
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=5, learning_rate=0.01,
                           experiment_name="Epochs = 5 (Short)")
    set3_results.append(result)
    
    # Experiment 3.2: Medium training (baseline)
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=10, learning_rate=0.01,
                           experiment_name="Epochs = 10 (Medium)")
    set3_results.append(result)
    
    # Experiment 3.3: Long training
    result = run_experiment(X_train, y_train, X_test, y_test,
                           hidden_size=128, epochs=20, learning_rate=0.01,
                           experiment_name="Epochs = 20 (Long)")
    set3_results.append(result)
    
    # Compare Set 3
    print("\n📈 Epoch Count Comparison:")
    compare_experiments(set3_results)
    all_results.extend(set3_results)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("🎯 ALL EXPERIMENTS SUMMARY")
    print("="*80)
    compare_experiments(all_results)
