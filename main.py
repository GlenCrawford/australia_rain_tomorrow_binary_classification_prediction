import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column
import numpy as np
import pandas as pd
import missingno as msno
import sklearn
from sklearn import preprocessing

### Load the input data ###

RAW_DATA_PATH = 'data_original.csv'
INPUT_DATA_PATH = 'data.csv'
INPUT_DATA_COLUMN_NAMES = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow']
INPUT_DATA_COLUMNS_TO_USE = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainTomorrow']

NUMERIC_COLUMNS_TO_SCALE = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

TEST_DATA_SET_SIZE = 1000
BATCH_SIZE = 5

LOCATION_COLUMN_CATEGORIES = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
DIRECTION_COLUMN_CATEGORIES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'] # Cardinal, intercardinal and secondary intercardinal directions.

# Make numpy values easier to read.
np.set_printoptions(precision = 3, suppress = True)

### Load the data ###

full_data_set = tf.data.experimental.make_csv_dataset(
  INPUT_DATA_PATH,
  batch_size = BATCH_SIZE,
  column_names = INPUT_DATA_COLUMN_NAMES,
  label_name = 'RainTomorrow',
  select_columns = INPUT_DATA_COLUMNS_TO_USE,
  header = True,
  num_epochs = 1,
  shuffle = True,
  shuffle_buffer_size = 100000000,
  ignore_errors = False
)

# Split the full data set up into two, one for training and one for testing.
test_data_set = full_data_set.take(TEST_DATA_SET_SIZE)
train_data_set = full_data_set.skip(TEST_DATA_SET_SIZE)

### Data preprocessing ###

# Use Pandas to do some basic preprocessing first.
def normalize_and_transform_input_data():
  data_frame = pd.read_csv(RAW_DATA_PATH, header = 0)

  # Remove rows with NaNs. Reduces from 142193 to 56420 rows. Some of the columns with NaNs have tens of thousands of them; too many to impute.
  data_frame = data_frame.dropna()

  # Column 'RainTomorrow' has discrete values 'No' and 'Yes'; map it to zero and one.
  data_frame['RainTomorrow'] = data_frame['RainTomorrow'].map({'No': 0, 'Yes': 1})

  # Scale/normalize numeric columns by calculating the z-score of each value.
  z_score_scaler = sklearn.preprocessing.StandardScaler(copy = True)
  data_frame[NUMERIC_COLUMNS_TO_SCALE] = z_score_scaler.fit_transform(data_frame[NUMERIC_COLUMNS_TO_SCALE].to_numpy())

  data_frame.to_csv(INPUT_DATA_PATH, na_rep = 'NA', index = False)

# Print what a single batch looks like.
def show_data_set_batch(data_set):
  for features, labels in data_set.take(1):
    for feature, values in features.items():
      print("{:9s}: {}".format(feature, values.numpy()))
    print('Labels:   ' + str(labels))

# Helper function to take a feature column and pass the example batch through it, in order to test preprocessing done in the feature column.
def pass_example_batch_through_feature_column(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print('\n\nFeature column result for example batch:')
  print(feature_layer(example_batch).numpy())

# Get a batch so we can look at some example values.
example_batch = next(iter(train_data_set))[0]

feature_columns = []

# Take a look at the example batch.
print('\nExample batch:')
show_data_set_batch(train_data_set)

# Go through each of the columns in the file, inspecting each one, discarding the ones that we don't want.
# For the ones that look to be possibly related to the label, build a feature column and preprocess as needed.
# (This is very repetitive, I deliberately did it the long way since this is a learning project.)

# Feature: Location. Categorical feature (example values: Sydney, Perth, Newcastle), use one-hot encoding.
location_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Location', LOCATION_COLUMN_CATEGORIES))
feature_columns.append(location_feature_column)

# Feature: MinTemp. Numeric, pre-scaled.
min_temp_feature_column = feature_column.numeric_column('MinTemp')
feature_columns.append(min_temp_feature_column)

# Feature: MaxTemp. Numeric, pre-scaled.
max_temp_feature_column = feature_column.numeric_column('MaxTemp')
feature_columns.append(max_temp_feature_column)

# Feature: Rainfall. Numeric, pre-scaled.
rainfall_feature_column = feature_column.numeric_column('Rainfall')
feature_columns.append(rainfall_feature_column)

# Feature: Evaporation. Numeric, pre-scaled.
evaporation_feature_column = feature_column.numeric_column('Evaporation')
feature_columns.append(evaporation_feature_column)

# Feature: Sunshine. Numeric, pre-scaled.
sunshine_feature_column = feature_column.numeric_column('Sunshine')
feature_columns.append(sunshine_feature_column)

# Feature: WindGustDir. Categorical feature (example values: E, SW, WNW), use one-hot encoding.
wind_gust_dir_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindGustDir', DIRECTION_COLUMN_CATEGORIES))
feature_columns.append(wind_gust_dir_feature_column)

# Feature: WindGustSpeed. Numeric, pre-scaled.
wind_gust_speed_feature_column = feature_column.numeric_column('WindGustSpeed')
feature_columns.append(wind_gust_speed_feature_column)

# Feature: WindDir9am. Categorical feature (example values: SE, NNW, WSW), use one-hot encoding.
wind_dir_9am_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindDir9am', DIRECTION_COLUMN_CATEGORIES))
feature_columns.append(wind_dir_9am_feature_column)

# Feature: WindDir3pm. Categorical feature (example values: NW, WSW, W), use one-hot encoding.
wind_dir_3pm_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindDir3pm', DIRECTION_COLUMN_CATEGORIES))
feature_columns.append(wind_dir_3pm_feature_column)

# Feature: WindSpeed9am. Numeric, pre-scaled.
wind_speed_9am_feature_column = feature_column.numeric_column('WindSpeed9am')
feature_columns.append(wind_speed_9am_feature_column)

# Feature: WindSpeed3pm. Numeric, pre-scaled.
wind_speed_3pm_feature_column = feature_column.numeric_column('WindSpeed3pm')
feature_columns.append(wind_speed_3pm_feature_column)

# Feature: Humidity9am. Numeric, pre-scaled.
humidity_9am_feature_column = feature_column.numeric_column('Humidity9am')
feature_columns.append(humidity_9am_feature_column)

# Feature: Humidity3pm. Numeric, pre-scaled.
humidity_3pm_feature_column = feature_column.numeric_column('Humidity3pm')
feature_columns.append(humidity_3pm_feature_column)

# Feature: Pressure9am. Numeric, pre-scaled.
pressure_9am_feature_column = feature_column.numeric_column('Pressure9am')
feature_columns.append(pressure_9am_feature_column)

# Feature: Pressure3pm. Numeric, pre-scaled.
pressure_3pm_feature_column = feature_column.numeric_column('Pressure3pm')
feature_columns.append(pressure_3pm_feature_column)

# Feature: Cloud9am. Numeric, pre-scaled.
cloud_9am_feature_column = feature_column.numeric_column('Cloud9am')
feature_columns.append(cloud_9am_feature_column)

# Feature: Cloud3pm. Numeric, pre-scaled.
cloud_3pm_feature_column = feature_column.numeric_column('Cloud3pm')
feature_columns.append(cloud_3pm_feature_column)

# Feature: Temp9am. Numeric, pre-scaled.
temp_9am_feature_column = feature_column.numeric_column('Temp9am')
feature_columns.append(temp_9am_feature_column)

# Feature: Temp3pm. Numeric, pre-scaled.
temp_3pm_feature_column = feature_column.numeric_column('Temp3pm')
feature_columns.append(temp_3pm_feature_column)

### Define the model ###

model = tf.keras.Sequential([
  tf.keras.layers.DenseFeatures(feature_columns),
  tf.keras.layers.Dense(16, activation = 'relu'),
  tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = ['binary_accuracy']
)

### Train the model ###

print('\nTraining:')

model.fit(
  train_data_set,
  epochs = 5
)

print('\nModel architecture:')
model.summary()

### Test the model ###

print('\nTesting:')

test_loss, test_accuracy = model.evaluate(
  test_data_set,
  verbose = 1
)

print('Test run loss: ' + str(test_loss))
print('Test run accuracy: ' + str(test_accuracy))
