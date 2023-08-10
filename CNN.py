%tensorflow_version 2.x  # this line is not required unless you are in a notebook
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

'''
CNN's are specialized images via Convolutions they highlight features
MaxPooling allows such features to be further highlighted, this continues
and then you flatten to transition to the DNN
'''

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images/255.0 - 0.5, test_images/255.0-0.5

plt.imshow(train_images[1])
plt.show()

model = models.Sequential(
    [
    #At the beginning, make sure to specify input_shape
    layers.Conv2D(32,3, activation = "relu", input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,3, activation = "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,3, activation = "relu"),
    layers.MaxPooling2D((2,2)),
    #Remember to seperate Convolution/Pooling layer from Dense
    layers.Flatten(),
    layers.Dense(32, activation = "relu"),
    layers.Dense(10)
    ]
)

model.summary()
model.compile("Adam",
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
model.fit(train_images, train_labels, batch_size = 32, epochs = 10)
model.predict(eval_images, eval_labels)