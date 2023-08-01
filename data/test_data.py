import numpy as np

# Import INPUT_SHAPE from federated_learning.py
from federated_learning import INPUT_SHAPE, NUM_CLASSES


def generate_test_data(num_samples, input_shape, num_classes):
    X = np.random.random((num_samples, input_shape))
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


NUM_TEST_SAMPLES = 1000

X_test, y_test = generate_test_data(NUM_TEST_SAMPLES, INPUT_SHAPE, NUM_CLASSES)
