import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import tool

tf.enable_eager_execution()

f = open('Images/list.txt')
class_names = list(map(lambda x: x[:-1], f.readlines()))
class_names.sort()

data_test, data_train = tool.load_data()

model = tf.keras.Sequential([
    # layers.BatchNormalization(),
    layers.Conv2D(32, (7, 7), input_shape=(60, 60, 3), activation='relu'),
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

model.compile(tf.train.AdamOptimizer(), tf.losses.softmax_cross_entropy)
def loss(model, x, y):
    print(x.shape)
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        lossVal = loss(model, inputs, targets)
    return lossVal, tape.gradient(lossVal, model.trainable_variables)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
global_step = tf.Variable(0)

from tensorflow import contrib
tfe = contrib.eager

train_loss_results = []
train_accuracy_results = []

num_epochs = 200

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    tmp1, tmp2 = data_train.get_batch(40)
    tmp1, tmp2 = tf.cast(tmp1, tf.float32), tf.cast(tmp2, tf.float32)
    for i in range(40):
        loss_value, grads = grad(model, np.expand_dims(tmp1[i], axis=0), np.expand_dims(tmp2[i], axis=0))
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    epoch_loss_avg(loss_value)
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 10 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

for i in range(10):
    tmp = data_train.get_batch(1000)
    model.fit(tmp[0], tmp[1])
