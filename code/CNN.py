import tensorflow as tf
from tensorflow.keras import layers
import tool
import os

os.chdir('..')
f = open('Images/list.txt')
class_names = list(map(lambda x: x[:-1], f.readlines()))
class_names.sort()

data_test, data_train = tool.load_data()

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

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        lossVal = loss(model, inputs, targets)
    return lossVal, tape.gradient(lossVal, model.trainable_variables)
    

for i in range(100):
    model.fit(data_train.get_batch())

