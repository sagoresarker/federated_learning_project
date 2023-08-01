import numpy as np
import tensorflow as tf
from data.test_data import generate_test_data
from models.server_model import create_server_model


# Import INPUT_SHAPE from federated_learning.py
from federated_learning import INPUT_SHAPE, NUM_CLASSES

# Create the server model
server = create_server_model(INPUT_SHAPE, NUM_CLASSES)

# Load the test data
NUM_TEST_SAMPLES = 1000
X_test, y_test = generate_test_data(NUM_TEST_SAMPLES, INPUT_SHAPE, NUM_CLASSES)

# Load the saved federated model
loaded_model = tf.keras.models.load_model("federated_learning_project/federated_model")

# Evaluate the model on the test data
logits = loaded_model(X_test, training=False)
predictions = tf.argmax(logits, axis=-1)
accuracy = np.mean(predictions.numpy() == y_test)
print(f"Test accuracy: {accuracy}")
