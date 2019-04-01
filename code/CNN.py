import tensorflow as tf
from tensorflow.keras import layers
import tool


model = tf.keras.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, (7, 7), input_shape=(60, 60, 1), activation='relu'),
    layers.Conv2D(64, (7, 7), activation='relu'),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.Conv2D(128, (5, 5), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(19, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit()