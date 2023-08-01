import tensorflow as tf
import numpy as np
from data.client_data import client_data
from models.server_model import create_server_model
from models.client_model import create_client_model

# Define the constants
NUM_ROUNDS = 10
LEARNING_RATE = 0.1
NUM_CLIENTS = 5
NUM_SAMPLES_PER_CLIENT = 1000
INPUT_SHAPE = 10
NUM_CLASSES = 2

# Create the models
server = create_server_model(INPUT_SHAPE, NUM_CLASSES)
client = create_client_model(INPUT_SHAPE, NUM_CLASSES)

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=LEARNING_RATE)  # Use the legacy optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def client_training(client_model, X, y):
    with tf.GradientTape() as tape:
        logits = client_model(X, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, client_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, client_model.trainable_weights))
    return client_model.get_weights()

for round_num in range(NUM_ROUNDS):
    server_weights = server.get_weights()
    client_weights = []

    for client_id in range(NUM_CLIENTS):
        client_model = create_client_model(INPUT_SHAPE, NUM_CLASSES)
        client_model.set_weights(server_weights)
        X_client, y_client = client_data[client_id]
        client_weights.append(client_training(client_model, X_client, y_client))

    average_weights = np.mean(client_weights, axis=0)
    server.set_weights(average_weights)

print("Federated learning training complete.")

# Save the trained federated model
server.save("federated_learning_project/federated_model")

print("Model saved.")
