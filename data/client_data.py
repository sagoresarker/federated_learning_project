import numpy as np

def generate_random_data(num_samples, input_shape, num_classes):
    X = np.random.random((num_samples, input_shape))
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

NUM_CLIENTS = 5
NUM_SAMPLES_PER_CLIENT = 1000
INPUT_SHAPE = 10
NUM_CLASSES = 2

client_data = [generate_random_data(NUM_SAMPLES_PER_CLIENT, INPUT_SHAPE, NUM_CLASSES) for _ in range(NUM_CLIENTS)]
