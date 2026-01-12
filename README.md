```bash
# output of neural_network.py
============================================================
Neural Network from Scratch - Testing
============================================================


============================================================
HYPERPARAMETER EXPERIMENTS
============================================================

Loading MNIST dataset...
Loading MNIST dataset...
Training set: 56000 samples
Test set: 14000 samples


############################################################
# EXPERIMENT SET 1: Learning Rate Impact
# (Keep hidden_size=128, epochs=10 constant)
############################################################

============================================================
🧪 LR = 0.001 (Very Small)
============================================================
Hidden size: 128, Epochs: 10, LR: 0.001

Epoch 1/10 - Loss: 0.6905 - Train Acc: 90.80% - Test Acc: 90.64%
Epoch 2/10 - Loss: 0.2908 - Train Acc: 92.82% - Test Acc: 92.73%
Epoch 3/10 - Loss: 0.2343 - Train Acc: 94.21% - Test Acc: 93.83%
Epoch 4/10 - Loss: 0.1960 - Train Acc: 95.04% - Test Acc: 94.76%
Epoch 5/10 - Loss: 0.1680 - Train Acc: 95.61% - Test Acc: 95.13%
Epoch 6/10 - Loss: 0.1474 - Train Acc: 96.18% - Test Acc: 95.55%
Epoch 7/10 - Loss: 0.1303 - Train Acc: 96.69% - Test Acc: 95.97%
Epoch 8/10 - Loss: 0.1169 - Train Acc: 97.06% - Test Acc: 96.24%
Epoch 9/10 - Loss: 0.1060 - Train Acc: 97.21% - Test Acc: 96.34%
Epoch 10/10 - Loss: 0.0964 - Train Acc: 97.49% - Test Acc: 96.49%

✅ Final Test Accuracy: 96.49%

============================================================
🧪 LR = 0.01 (Sweet Spot)
============================================================
Hidden size: 128, Epochs: 10, LR: 0.01

Epoch 1/10 - Loss: 0.2721 - Train Acc: 96.12% - Test Acc: 95.21%
Epoch 2/10 - Loss: 0.1072 - Train Acc: 97.63% - Test Acc: 96.64%
Epoch 3/10 - Loss: 0.0753 - Train Acc: 98.08% - Test Acc: 96.98%
Epoch 4/10 - Loss: 0.0580 - Train Acc: 98.71% - Test Acc: 97.24%
Epoch 5/10 - Loss: 0.0450 - Train Acc: 98.81% - Test Acc: 97.26%
Epoch 6/10 - Loss: 0.0372 - Train Acc: 98.43% - Test Acc: 96.81%
Epoch 7/10 - Loss: 0.0274 - Train Acc: 99.40% - Test Acc: 97.64%
Epoch 8/10 - Loss: 0.0239 - Train Acc: 99.64% - Test Acc: 97.80%
Epoch 9/10 - Loss: 0.0170 - Train Acc: 99.37% - Test Acc: 97.59%
Epoch 10/10 - Loss: 0.0148 - Train Acc: 99.72% - Test Acc: 97.74%

✅ Final Test Accuracy: 97.74%


📈 Learning Rate Comparison:

================================================================================
📊 EXPERIMENT COMPARISON
================================================================================
Experiment                Hidden     LR         Epochs     Final Acc    Final Loss
--------------------------------------------------------------------------------
Experiment 1              128        0.001      10         96.49%       0.0964
Experiment 2              128        0.01       10         97.74%       0.0148
================================================================================


############################################################
# EXPERIMENT SET 2: Hidden Layer Size Impact
# (Keep lr=0.01, epochs=10 constant)
############################################################

============================================================
🧪 Hidden = 64 (Small)
============================================================
Hidden size: 64, Epochs: 10, LR: 0.01

Epoch 1/10 - Loss: 0.2910 - Train Acc: 96.00% - Test Acc: 95.36%
Epoch 2/10 - Loss: 0.1220 - Train Acc: 96.67% - Test Acc: 95.60%
Epoch 3/10 - Loss: 0.0901 - Train Acc: 97.95% - Test Acc: 96.62%
Epoch 4/10 - Loss: 0.0738 - Train Acc: 98.24% - Test Acc: 96.83%
Epoch 5/10 - Loss: 0.0614 - Train Acc: 98.59% - Test Acc: 96.81%
Epoch 6/10 - Loss: 0.0538 - Train Acc: 98.11% - Test Acc: 96.45%
Epoch 7/10 - Loss: 0.0455 - Train Acc: 98.75% - Test Acc: 96.94%
Epoch 8/10 - Loss: 0.0390 - Train Acc: 98.57% - Test Acc: 96.72%
Epoch 9/10 - Loss: 0.0355 - Train Acc: 98.84% - Test Acc: 96.83%
Epoch 10/10 - Loss: 0.0283 - Train Acc: 99.32% - Test Acc: 97.26%

✅ Final Test Accuracy: 97.26%

============================================================
🧪 Hidden = 128 (Medium)
============================================================
Hidden size: 128, Epochs: 10, LR: 0.01

Epoch 1/10 - Loss: 0.2721 - Train Acc: 96.12% - Test Acc: 95.21%
Epoch 2/10 - Loss: 0.1072 - Train Acc: 97.63% - Test Acc: 96.64%
Epoch 3/10 - Loss: 0.0753 - Train Acc: 98.08% - Test Acc: 96.98%
Epoch 4/10 - Loss: 0.0580 - Train Acc: 98.71% - Test Acc: 97.24%
Epoch 5/10 - Loss: 0.0450 - Train Acc: 98.81% - Test Acc: 97.26%
Epoch 6/10 - Loss: 0.0372 - Train Acc: 98.43% - Test Acc: 96.81%
Epoch 7/10 - Loss: 0.0274 - Train Acc: 99.40% - Test Acc: 97.64%
Epoch 8/10 - Loss: 0.0239 - Train Acc: 99.64% - Test Acc: 97.80%
Epoch 9/10 - Loss: 0.0170 - Train Acc: 99.37% - Test Acc: 97.59%
Epoch 10/10 - Loss: 0.0148 - Train Acc: 99.72% - Test Acc: 97.74%


============================================================
🧪 Hidden = 256 (Large)
============================================================
Hidden size: 256, Epochs: 10, LR: 0.01

Epoch 1/10 - Loss: 0.2599 - Train Acc: 96.17% - Test Acc: 95.29%
Epoch 2/10 - Loss: 0.0989 - Train Acc: 98.00% - Test Acc: 96.84%
Epoch 3/10 - Loss: 0.0668 - Train Acc: 98.61% - Test Acc: 97.34%
Epoch 4/10 - Loss: 0.0504 - Train Acc: 98.92% - Test Acc: 97.51%
Epoch 5/10 - Loss: 0.0376 - Train Acc: 98.66% - Test Acc: 97.21%
Epoch 6/10 - Loss: 0.0285 - Train Acc: 99.58% - Test Acc: 97.79%
Epoch 7/10 - Loss: 0.0205 - Train Acc: 99.29% - Test Acc: 97.36%
Epoch 8/10 - Loss: 0.0139 - Train Acc: 99.78% - Test Acc: 97.81%
Epoch 9/10 - Loss: 0.0099 - Train Acc: 99.92% - Test Acc: 98.00%
Epoch 10/10 - Loss: 0.0065 - Train Acc: 99.83% - Test Acc: 97.95%

✅ Final Test Accuracy: 97.95%

📈 Hidden Size Comparison:

================================================================================
📊 EXPERIMENT COMPARISON
================================================================================
Experiment                Hidden     LR         Epochs     Final Acc    Final Loss
--------------------------------------------------------------------------------
Experiment 1              64         0.01       10         97.26%       0.0283
Experiment 2              128        0.01       10         97.74%       0.0148
Experiment 3              256        0.01       10         97.95%       0.0065
================================================================================

############################################################
# EXPERIMENT SET 3: Training Duration Impact
# (Keep hidden_size=128, lr=0.01 constant)
############################################################

============================================================
🧪 Epochs = 5 (Short)
============================================================
Hidden size: 128, Epochs: 5, LR: 0.01

Epoch 1/5 - Loss: 0.2721 - Train Acc: 96.12% - Test Acc: 95.21%
Epoch 2/5 - Loss: 0.1072 - Train Acc: 97.63% - Test Acc: 96.64%
Epoch 3/5 - Loss: 0.0753 - Train Acc: 98.08% - Test Acc: 96.98%
Epoch 4/5 - Loss: 0.0580 - Train Acc: 98.71% - Test Acc: 97.24%
Epoch 5/5 - Loss: 0.0450 - Train Acc: 98.81% - Test Acc: 97.26%

✅ Final Test Accuracy: 97.26%

============================================================
🧪 Epochs = 10 (Medium)
============================================================
Hidden size: 128, Epochs: 10, LR: 0.01

Epoch 1/10 - Loss: 0.2721 - Train Acc: 96.12% - Test Acc: 95.21%
Epoch 2/10 - Loss: 0.1072 - Train Acc: 97.63% - Test Acc: 96.64%
Epoch 3/10 - Loss: 0.0753 - Train Acc: 98.08% - Test Acc: 96.98%
Epoch 4/10 - Loss: 0.0580 - Train Acc: 98.71% - Test Acc: 97.24%
Epoch 5/10 - Loss: 0.0450 - Train Acc: 98.81% - Test Acc: 97.26%
Epoch 6/10 - Loss: 0.0372 - Train Acc: 98.43% - Test Acc: 96.81%
Epoch 7/10 - Loss: 0.0274 - Train Acc: 99.40% - Test Acc: 97.64%
Epoch 8/10 - Loss: 0.0239 - Train Acc: 99.64% - Test Acc: 97.80%
Epoch 9/10 - Loss: 0.0170 - Train Acc: 99.37% - Test Acc: 97.59%
Epoch 10/10 - Loss: 0.0148 - Train Acc: 99.72% - Test Acc: 97.74%

✅ Final Test Accuracy: 97.74%

============================================================
🧪 Epochs = 20 (Long)
============================================================
Hidden size: 128, Epochs: 20, LR: 0.01

Epoch 1/20 - Loss: 0.2721 - Train Acc: 96.12% - Test Acc: 95.21%
Epoch 2/20 - Loss: 0.1072 - Train Acc: 97.63% - Test Acc: 96.64%
Epoch 3/20 - Loss: 0.0753 - Train Acc: 98.08% - Test Acc: 96.98%
Epoch 4/20 - Loss: 0.0580 - Train Acc: 98.71% - Test Acc: 97.24%
Epoch 5/20 - Loss: 0.0450 - Train Acc: 98.81% - Test Acc: 97.26%
Epoch 6/20 - Loss: 0.0372 - Train Acc: 98.43% - Test Acc: 96.81%
Epoch 7/20 - Loss: 0.0274 - Train Acc: 99.40% - Test Acc: 97.64%
Epoch 8/20 - Loss: 0.0239 - Train Acc: 99.64% - Test Acc: 97.80%
Epoch 9/20 - Loss: 0.0170 - Train Acc: 99.37% - Test Acc: 97.59%
Epoch 10/20 - Loss: 0.0148 - Train Acc: 99.72% - Test Acc: 97.74%
Epoch 11/20 - Loss: 0.0111 - Train Acc: 99.85% - Test Acc: 97.84%
Epoch 12/20 - Loss: 0.0066 - Train Acc: 99.81% - Test Acc: 97.81%
Epoch 13/20 - Loss: 0.0050 - Train Acc: 99.96% - Test Acc: 97.93%
Epoch 14/20 - Loss: 0.0031 - Train Acc: 99.99% - Test Acc: 98.04%
Epoch 15/20 - Loss: 0.0023 - Train Acc: 100.00% - Test Acc: 98.05%
Epoch 16/20 - Loss: 0.0015 - Train Acc: 99.99% - Test Acc: 97.98%
Epoch 17/20 - Loss: 0.0012 - Train Acc: 99.98% - Test Acc: 98.01%
Epoch 18/20 - Loss: 0.0010 - Train Acc: 100.00% - Test Acc: 98.09%
Epoch 19/20 - Loss: 0.0009 - Train Acc: 100.00% - Test Acc: 98.11%
Epoch 20/20 - Loss: 0.0008 - Train Acc: 100.00% - Test Acc: 98.11%

✅ Final Test Accuracy: 98.11%

📈 Epoch Count Comparison:

================================================================================
📊 EXPERIMENT COMPARISON
================================================================================
Experiment                Hidden     LR         Epochs     Final Acc    Final Loss
--------------------------------------------------------------------------------
Experiment 1              128        0.01       5          97.26%       0.0450
Experiment 2              128        0.01       10         97.74%       0.0148
Experiment 3              128        0.01       20         98.11%       0.0008
================================================================================


================================================================================
🎯 ALL EXPERIMENTS SUMMARY
================================================================================

================================================================================
📊 EXPERIMENT COMPARISON
================================================================================
Experiment                Hidden     LR         Epochs     Final Acc    Final Loss
--------------------------------------------------------------------------------
Experiment 1              128        0.001      10         96.49%       0.0964
Experiment 2              128        0.01       10         97.74%       0.0148
Experiment 3              64         0.01       10         97.26%       0.0283
Experiment 4              128        0.01       10         97.74%       0.0148
Experiment 5              256        0.01       10         97.95%       0.0065
Experiment 6              128        0.01       5          97.26%       0.0450
Experiment 7              128        0.01       10         97.74%       0.0148
Experiment 8              128        0.01       20         98.11%       0.0008
================================================================================
```