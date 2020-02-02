# Binary classification machine learning model to predict whether it will rain tomorrow in Australia.

This is a Tensorflow 2 and Keras neural network that uses binary classification to predict whether, given meteorological observations of a given day at a given weather station in Australia, it will rain there the next day. The model is trained and tested on a dataset containing about 10 years of daily weather observations from numerous Australian weather stations.

The model currently has an accuracy of approximately 87%. Given that it doesn't rain exactly 50% of days, there are a lot more rows in the dataset where the target "RainTomorrow" column has a "No" value than "Yes". This means that you can make a complete guess and be right by random chance about 70% of the time. My goal was therefore to get the model accuracy to somewhere around 90%.

Here is the structure of the dataset used for training and testing, showing the header and two data rows:

| Date       | Location | MinTemp | MaxTemp | Rainfall | Evaporation | Sunshine | WindGustDir | WindGustSpeed | WindDir9am | WindDir3pm | WindSpeed9am | WindSpeed3pm | Humidity9am | Humidity3pm | Pressure9am | Pressure3pm | Cloud9am | Cloud3pm | Temp9am | Temp3pm | RainToday | RISK_MM | RainTomorrow |
|:----------:|:--------:|:-------:|:-------:|:--------:|:-----------:|:--------:|:-----------:|:-------------:|:----------:|:----------:|:------------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:--------:|:--------:|:-------:|:-------:|:---------:|:-------:|:------------:|
| 2010-10-20 | Sydney   | 12.9    | 20.3    | 0.2      | 3           | 10.9     | ENE         | 37            | W          | E          | 11           | 26           | 70          | 57          | 1028.8      | 1025.6      | 3        | 1        | 16.9    | 19.8    | No        | 0       | No           |
| 2017-06-25 | Brisbane | 11      | 24.2    | 0        | 2.2         | 9.8      | ENE         | 20            | SSW        | NNE        | 2            | 7            | 68          | 53          | 1020.5      | 1017.3      | 6        | 3        | 15.9    | 22.6    | No        | 0       | Yes          |

The data was sourced from this [Kaggle dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) compiled by Joe Young and Adam Young, which was in turn sourced from [http://www.bom.gov.au/climate/data](http://www.bom.gov.au/climate/data) and [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/). This data is available under a Creative Commons (CC) Attribution 3.0 licence. For details on the meaning of each observation, see [this page](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml). Copyright Commonwealth of Australia, Bureau of Meteorology.

## Requirements

* Python (developed with version 3.7.4).

* See dependencies.txt for packages and versions (and below to install).

## Data preprocessing

Data preprocessing is done by a combination of Pandas (to drop NaN rows and map Yes/No strings into 1/0 binary integers), scikit-learn (to scale/normalize numeric columns by calculating the z-score of each of their values), and Tensorflow to apply one-hot encoding to categorical columns. The model's input layer is thus a combination of pre-normalized numeric columns and one-hot encoded categorical columns.

This is done as a two-step process to speed up the runtime: when required, the `normalize_and_transform_input_data` method reads the CSV with the original dataset, applies the preprocessing transformations, and exports the result to a new CSV file. Subsequent executions of the file simply read the latter file, rather than run the preprocessing every time.

The following columns were skipped and not used as features for the model; all the rest were used:

* __Date:__ Not relevant.

* __RainToday:__ This is just a boolean representation of the numeric column "Rainfall". Experimented with adding this feature to the model, but had no effect on accuracy.

* __RISK_MM:__ This is the amount of rain for the following day. This was used to create the label/target column "RainTomorrow". This would be used if the model was doing regression, rather than classification.

* __RainTomorrow:__ Used as the training label/target.

The output of the model is just a single sigmoid-activation neuron which predicts target variable "RainTomorrow".

## Setup

Clone the Git repository.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

Download the dataset from the source above, and save in the project directory as `data_original.csv`.

## Run

```bash
python -W ignore main.py
```

## Monitoring/logging

After training, run:

```
$ tensorboard --logdir logs/fit
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.0.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then open the above URL in your browser to view the model in TensorBoard.
