"""Provide a neural network for EMNIST balanced character classification.

Architecture:
- Input: 784 (28x28 flattened)
- Hidden Layer 1: 2048 units with ReLU and dropout 0.1
- Hidden Layer 2: 1024 units with ReLU and dropout 0.2
- Hidden Layer 3: 512 units with ReLU and dropout 0.2
- Output Layer: 47 units with Softmax
"""
