import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column

import data_preprocessing

TEST_DATA_SET_SIZE = 1000
BATCH_SIZE = 5

LOG_DIRECTORY = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

data_preprocessing.normalize_and_transform_input_data()

### Load the preprocessed data ###

# Print what a single batch looks like.
def show_data_set_batch(data_set):
  for features, labels in data_set.take(1):
    for feature, values in features.items():
      print("{:9s}: {}".format(feature, values.numpy()))
    print('Labels:   ' + str(labels))

full_data_set = tf.data.experimental.make_csv_dataset(
  data_preprocessing.INPUT_DATA_PATH,
  batch_size = BATCH_SIZE,
  column_names = data_preprocessing.INPUT_DATA_COLUMN_NAMES,
  label_name = 'RainTomorrow',
  select_columns = data_preprocessing.INPUT_DATA_COLUMNS_TO_USE,
  header = True,
  num_epochs = 1,
  shuffle = True,
  shuffle_buffer_size = 100000000,
  ignore_errors = False
)

# Split the full data set up into two, one for training and one for testing.
test_data_set = full_data_set.take(TEST_DATA_SET_SIZE)
train_data_set = full_data_set.skip(TEST_DATA_SET_SIZE)

# Get a batch so we can look at some example values.
example_batch = next(iter(train_data_set))[0]

feature_columns = []

# Take a look at the example batch.
print('\nExample batch:')
show_data_set_batch(train_data_set)

### Build the feature columns for the input layer ###

# Helper function to take a feature column and pass the example batch through it, in order to test preprocessing done in the feature column.
def pass_example_batch_through_feature_column(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print('\n\nFeature column result for example batch:')
  print(feature_layer(example_batch).numpy())

# Go through each of the columns in the file, inspecting each one, discarding the ones that we don't want.
# For the ones that look to be possibly related to the label, build a feature column and preprocess as needed.
# (This is very repetitive, I deliberately did it the long way since this is a learning project.)

# Feature: Location. Categorical feature (example values: Sydney, Perth, Newcastle), use one-hot encoding.
location_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Location', data_preprocessing.LOCATION_COLUMN_CATEGORIES))
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
wind_gust_dir_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindGustDir', data_preprocessing.DIRECTION_COLUMN_CATEGORIES))
feature_columns.append(wind_gust_dir_feature_column)

# Feature: WindGustSpeed. Numeric, pre-scaled.
wind_gust_speed_feature_column = feature_column.numeric_column('WindGustSpeed')
feature_columns.append(wind_gust_speed_feature_column)

# Feature: WindDir9am. Categorical feature (example values: SE, NNW, WSW), use one-hot encoding.
wind_dir_9am_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindDir9am', data_preprocessing.DIRECTION_COLUMN_CATEGORIES))
feature_columns.append(wind_dir_9am_feature_column)

# Feature: WindDir3pm. Categorical feature (example values: NW, WSW, W), use one-hot encoding.
wind_dir_3pm_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('WindDir3pm', data_preprocessing.DIRECTION_COLUMN_CATEGORIES))
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

# Feature: RainToday. Boolean feature (values: 0 and 1), treat it as categorical, use one-hot encoding.
# This is just a boolean representation of the "Rainfall" column. Removed since it has no additional effect on model accuracy.
# rain_today_feature_column = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('RainToday', data_preprocessing.BOOLEAN_COLUMN_CATEGORIES))
# feature_columns.append(rain_today_feature_column)

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
  epochs = 5,
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir = LOG_DIRECTORY)]
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
