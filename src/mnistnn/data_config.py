"""Define file paths for the EMNIST training and test datasets."""

import os
from os.path import join

#
# Set file paths based on added EMNIST Datasets
#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "..", "data")

# Training data file paths dictionary
dict_config_train: dict[str, str] = {
    'input_path': input_path,
    'training_images_filepath': join(
        input_path, 'emnist-balanced-train-images-idx3-ubyte/emnist-balanced-train-images-idx3-ubyte'
    ),
    'training_labels_filepath': join(
        input_path, 'emnist-balanced-train-labels-idx1-ubyte/emnist-balanced-train-labels-idx1-ubyte'
    ),
}

# Test data file paths dictionary
dict_config_test: dict[str, str] = {
    'input_path': input_path,
    'test_images_filepath': join(
        input_path, 'emnist-balanced-test-images-idx3-ubyte/emnist-balanced-test-images-idx3-ubyte'
    ),
    'test_labels_filepath': join(
        input_path, 'emnist-balanced-test-labels-idx1-ubyte/emnist-balanced-test-labels-idx1-ubyte'
    ),
}

# Export dictionaries for use in other Python files
__all__: list[str] = ['dict_config_train', 'dict_config_test']
