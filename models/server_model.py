import tensorflow as tf

def create_server_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(num_classes)
    ])
    return model
