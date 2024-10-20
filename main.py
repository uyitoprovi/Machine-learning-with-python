import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Correctly shape the input data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float).reshape(-1, 1)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction (convert list to numpy array)
print(model.predict(np.array([[10.0]])))

# Print learned weights
print("Here are the learned weights: {}".format(model.layers[0].get_weights()))
