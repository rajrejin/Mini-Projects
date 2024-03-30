from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Define the model
model = tf.keras.Sequential([
    # Stage 1
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # Stage 2
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # Stage 3
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Stage 4
    layers.Flatten(),
    # Stage 5
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    # Stage 6
    layers.Dense(10, activation='softmax')
])

# Display the model summary
model.summary()

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to be values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to be 28x28x1 (required for Conv2D layers)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 10 epochs and capture the history
history = model.fit(train_images, train_labels, epochs=10)

# Plot the training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy
print(f'Test accuracy: {test_accuracy}')