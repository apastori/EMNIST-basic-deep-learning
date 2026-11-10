"""Provide a neural network for EMNIST balanced character classification.

Architecture:
- Input: 784 (28x28 flattened)
- Hidden Layer 1: 512 units with ReLU and dropout 0.2
- Hidden Layer 2: 256 units with ReLU and dropout 0.3
- Hidden Layer 3: 128 units with ReLU and dropout 0.3
- Output Layer: 47 units with Softmax
"""
