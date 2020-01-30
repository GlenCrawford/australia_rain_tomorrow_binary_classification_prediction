import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column

import numpy as np
import pandas as pd

### Load the input data ###

INPUT_DATA_PATH = 'data.csv'
INPUT_DATA_COLUMN_NAMES = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow']
INPUT_DATA_COLUMNS_TO_USE = ['Location', 'RainTomorrow']

# Note: Column 'RainTomorrow' has discrete values 'No' and 'Yes'; map it to zero and one.

# Columns to go through:
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

BATCH_SIZE = 5 # 500

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

### Load the data ###

data_set = tf.data.experimental.make_csv_dataset(
  INPUT_DATA_PATH,
  batch_size = BATCH_SIZE,
  column_names = INPUT_DATA_COLUMN_NAMES,
  label_name = 'RainTomorrow',
  select_columns = INPUT_DATA_COLUMNS_TO_USE,
  header = True,
  num_epochs = 1,
  shuffle = True, # True
  shuffle_buffer_size = 1000000, # 100000000
  ignore_errors = False
)

### Data preprocessing ###

# Print what a single batch looks like.
def show_data_set_batch():
  for features, labels in data_set.take(1):
    for feature, values in features.items():
      print("{:9s}: {}".format(feature, values.numpy()))
    print('Labels:   ' + str(labels))

# Helper function to take a feature column and pass the example batch through it, in order to test preprocessing done in the feature column.
def pass_example_batch_through_feature_column(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print('\n\nFeature column result for example batch:')
  print(feature_layer(example_batch).numpy())

# # Function to normalize numeric data based on the specified mean and standard deviation of the column.
# def normalize_numeric_data(data, mean, std):
#   return (data - mean) / std

# Get a batch so we can look at some example values.
example_batch = next(iter(data_set))[0]

feature_columns = []

# Take a look at the example batch.
print('\nExample batch:')
show_data_set_batch()

# Feature: Location. Categorical feature (example values: Sydney, Perth, Newcastle), use one-hot encoding.
LOCATION_COLUMN_CATEGORIES = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
location_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Location', LOCATION_COLUMN_CATEGORIES))
feature_columns.append(location_feature_column)
pass_example_batch_through_feature_column(location_feature_column)
