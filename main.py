import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column

import numpy as np
import pandas as pd

### Load the input data ###

INPUT_DATA_PATH = 'data.csv'
INPUT_DATA_COLUMN_NAMES = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow']
INPUT_DATA_COLUMNS_TO_USE = ['Location', 'RainTomorrow']

# 'MinTemp'
# 'MaxTemp'
# 'Rainfall'
# 'Evaporation'
# 'Sunshine'
# 'WindGustDir'
# 'WindGustSpeed'
# 'WindDir9am'
# 'WindDir3pm'
# 'WindSpeed9am'
# 'WindSpeed3pm'
# 'Humidity9am'
# 'Humidity3pm'
# 'Pressure9am'
# 'Pressure3pm'
# 'Cloud9am'
# 'Cloud3pm'
# 'Temp9am'
# 'Temp3pm'
# 'RainToday'

LOCATION_COLUMN_CATEGORIES = ['?']

BATCH_SIZE = 5 # 500

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

### Load the data ###

data_frame = pd.read_csv(INPUT_DATA_PATH, header = 0, names = INPUT_DATA_COLUMN_NAMES, usecols = INPUT_DATA_COLUMNS_TO_USE)

# Remove rows with missing values.
data_frame = data_frame.dropna()

# print(data_frame.describe())
# print(data_frame['WindDir9am'].unique())

# Load the Pandas data frame into a Tensorflow data set.
label_column = data_frame.pop('RainTomorrow')
data_set = tf.data.Dataset.from_tensor_slices((data_frame.values, label_column.values))

### Data preprocessing ###

def show_batch(batch):
  for features, label in batch:
    for feature in features:
      print('Feature: ' + str(feature))
    print('Label: ' + str(label) + '\n')

# Helper function to take a feature column and pass the example batch through it, in order to test preprocessing done in the feature column.
def pass_example_batch_through_feature_column(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

# Function to normalize numeric data based on the specified mean and standard deviation of the column.
def normalize_numeric_data(data, mean, std):
  return (data - mean) / std

# Get a batch so we can look at some example values.
example_batch = data_set.shuffle(len(data_frame)).take(BATCH_SIZE)

# Take a look at the example batch.
print('\n\nExample batch:')
show_batch(example_batch)

# Now let's go through a batch (where batch size is 5) and look at each of the features and their values, and test feature columns to preprocess them.

# Location feature (raw):



# Location feature (preprocessed):
# Categorical feature, use one-hot encoding.
print('\n\nLocation feature (preprocessed):')
pass_example_batch_through_feature_column(feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Location', LOCATION_COLUMN_CATEGORIES)))
# ???


# data_set_batches = data_set.shuffle(len(data_frame)).batch(BATCH_SIZE)
