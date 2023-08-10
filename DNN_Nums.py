#!pip install tensorflow keras numpy mnist matplotlib
'''DNN Classifier as opposed to a linear classifier, is less flexible than
a Convolution Neural Network, but powerful for analyzing trends
in image/non-image data-set'''

import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#Importing dataset from mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()
eval_images = mnist.test_images()
eval_labels = mnist.test_labels()
print(type(train_labels), type(eval_labels))


#convert images to (-0.5, 0.5) for better processing
train_images = (train_images/255.0) - 0.5
eval_images = (eval_images/255.0)-0.5
#Flatten 2D images
print(train_images.size)
train_images = np.reshape(train_images,(-1, 784))
eval_images = np.reshape(eval_images,(-1, 784))
print(train_images.shape)
print(eval_images.shape)

#Path1 for creating model, manually building sequential model
model = keras.Sequential([
    layers.Dense(64, activation = "relu", input_dim = 784),
    layers.Dense(64, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])

model.compile(optimizer= "Adam",
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(
    train_images,
      to_categorical(train_labels),
      epochs = 5,
      batch_size = 32
)

evaluate = model.evaluate(
    eval_images,
    to_categorical(eval_labels),
    batch_size = 32
)
print("accuracy =",evaluate[1])


#Print Results of prediction
result = model.predict(eval_images[:5])
for i in range(0, 5):
  plt.imshow(np.reshape(eval_images[i], (28,28)), cmap = "gray")
  plt.show()
print(np.argmax(result, axis = 1))
print(eval_labels[:5])


