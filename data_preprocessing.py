import numpy as np
import pandas as pd
import missingno as msno
import sklearn
from sklearn import preprocessing

RAW_DATA_PATH = 'raw_data.csv'
INPUT_DATA_PATH = 'data.csv'
INPUT_DATA_COLUMN_NAMES = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow']
INPUT_DATA_COLUMNS_TO_USE = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']

NUMERIC_COLUMNS_TO_SCALE = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

LOCATION_COLUMN_CATEGORIES = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
DIRECTION_COLUMN_CATEGORIES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'] # Cardinal, intercardinal and secondary intercardinal directions.
BOOLEAN_COLUMN_CATEGORIES = [0, 1]

# Make numpy values easier to read.
np.set_printoptions(precision = 3, suppress = True)

# Use Pandas to do some basic preprocessing.
def normalize_and_transform_input_data():
  data_frame = pd.read_csv(RAW_DATA_PATH, header = 0)

  # Remove rows with NaNs. Reduces from 142193 to 56420 rows. Some of the columns with NaNs have tens of thousands of them; too many to impute.
  data_frame = data_frame.dropna()

  # Columns 'RainToday' and 'RainTomorrow' have discrete values 'No' and 'Yes'; map them to zero and one.
  data_frame['RainToday'] = data_frame['RainToday'].map({'No': 0, 'Yes': 1})
  data_frame['RainTomorrow'] = data_frame['RainTomorrow'].map({'No': 0, 'Yes': 1})

  # Scale/normalize numeric columns by calculating the z-score of each value.
  z_score_scaler = sklearn.preprocessing.StandardScaler(copy = True)
  data_frame[NUMERIC_COLUMNS_TO_SCALE] = z_score_scaler.fit_transform(data_frame[NUMERIC_COLUMNS_TO_SCALE].to_numpy())

  data_frame.to_csv(INPUT_DATA_PATH, na_rep = 'NA', index = False)
