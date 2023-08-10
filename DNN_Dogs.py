#!pip install sklearn

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pathlib
'''DNN Classifier as opposed to a linear classifier, is less flexible than
a Convolution Neural Network, but powerful for analyzing trends
in image/non-image data-set'''

#Importing data, and initially processing/parsing data
#Including: removing 'description' + streamlining target margins
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                        extract=True, cache_dir='.')
dataframe = pd.read_csv(csv_file)

dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4,0,1)
dataframe = dataframe.drop(columns = ['AdoptionSpeed', 'Description'])

train_set, eval_set = train_test_split(dataframe, test_size = 0.1)
eval_set, test_set = train_test_split(eval_set, test_size = 0.3)

train_label, eval_label, test_label = train_set.pop("target"), eval_set.pop("target"), test_set.pop("target")


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

#Testing Features
for feature_batch, label_batch in train_ds.take(1):
  print("Every feature", list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print("A batch of targets: ", label_batch)

NUMERIC_COLUMN = ["Age", "Fee", "PhotoAmt"]
CATEGORICAL_COLUMNS = ["Type", "Breed1","Gender", "Color1", "Color2", "MaturitySize", "FurLength", "Vaccinated", "Sterilized", "Health"]
# EMBEDDING_COLUMNS = ["Breed1"]

feature_columns = []

# for key in EMBEDDING_COLUMNS:
#   breed1 = feature_column.categorical_column_with_vocabulary_list(
#       col_name, dataframe[col_name].unique()
#   )


for key in CATEGORICAL_COLUMNS:
  vocabulary = train_set[key].unique()
  categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary)
  indicator_column = tf.feature_column.indicator_column(categorical_column)
  #Remember to ad tf.feature_column.indicator_column
  feature_columns.append(indicator_column)
  #feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary))

for key in NUMERIC_COLUMN:
  feature_columns.append(tf.feature_column.numeric_column(key=key, dtype = tf.float32))
print(feature_columns)

'''Path 1
#Creating Data Set
batch_size = 32
train_ds = input_fn(train_set, train_label, batch_size=batch_size)
val_ds = input_fn(eval_set, eval_label, training=False, batch_size=batch_size)
test_ds = input_fn(test_set, test_label, training=False, batch_size=batch_size)

#The normal-streamlined way requires feature_columns, this_way
requires first layer to use feature_columns

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          steps_per_epoch = 32
          )
loss, accuracy = model.evaluate(val_ds)
print("Accuracy", accuracy)

predictions = model.predict(test_ds)
#print(type(test_set))
print(test_label)
for i in range(0, 2):
  print(test_label[i:i+1])
  print("predictions ",predictions[i], "\n")
'''


'''
Path2
# REMEMBER THAT tf.estimator is it's own class utilizing 
    train/evaluate/predict
 Whereas tf.keras.Sequential:
    compile/fit/evaluate
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns= feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 30, 30],
    # The model must choose between 3 classes.
    n_classes=5)
    
classifier.train(
    input_fn=lambda: input_fn(train_set, train_label, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(eval_set, eval_label, training=False)
)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

'''
