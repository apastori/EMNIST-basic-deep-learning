"""Provide a neural network for EMNIST balanced character classification.

Architecture:
- Input: 784 (28x28 flattened)
- Hidden Layer 1: 1024 units with ReLU and dropout 0.15
- Hidden Layer 2: 512 units with ReLU and dropout 0.25
- Hidden Layer 3: 256 units with ReLU and dropout 0.25
- Output Layer: 47 units with Softmax
"""
