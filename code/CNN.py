import tensorflow as tf
from tensorflow.keras import layers
import os

os.chdir('..')

model = tf.keras.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, (16, 16), input_shape=(600, 600, 1), activation='relu'),
    layers.Conv2D(64, (12, 12), activation='relu'),
    layers.Conv2D(64, (10, 10), activation='relu'),
    layers.Conv2D(128, (8, 8), activation='relu'),
    layers.Conv2D(128, (4, 4), activation='relu'),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(19, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



