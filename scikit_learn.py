import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

import data_preprocessing

TEST_DATA_SET_SIZE = 1000
BATCH_SIZE = 200

# Perform the data preprocessing common with the TensorFlow implementation.
data_preprocessing.normalize_and_transform_input_data()

full_data_set = pd.read_csv(
  data_preprocessing.INPUT_DATA_PATH,
  header = 0,
  usecols = data_preprocessing.INPUT_DATA_COLUMNS_TO_USE
)

labels = full_data_set.pop('RainTomorrow').values

# Apply one-hot encoding to categorical columns.
full_data_set = pd.get_dummies(
  full_data_set,
  columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],
  sparse = False
)

# Split the full data set up into two, one for training and one for testing.
train_data_set, test_data_set, train_labels, test_labels = sklearn.model_selection.train_test_split(
  full_data_set,
  labels,
  test_size = TEST_DATA_SET_SIZE,
  shuffle = True
)

# Configure a model.
model = sklearn.neural_network.MLPClassifier(
  hidden_layer_sizes = (16),
  activation = 'relu',
  solver = 'adam',
  alpha = 1e-5,
  batch_size = BATCH_SIZE,
  learning_rate = 'constant',
  verbose = True
)

# Train the model.
model.fit(train_data_set, train_labels)

# Test the model.
test_result = model.score(test_data_set, test_labels)
print('Test run mean accuracy: ' + str(round(test_result, 4)))
