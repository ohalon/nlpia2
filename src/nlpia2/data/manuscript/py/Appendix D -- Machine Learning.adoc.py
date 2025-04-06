from keras.models import Sequential

from keras.layers import Dropout, LSTM, Flatten, Dense

num_neurons = 20  # <1>

maxlen = 100

embedding_dims = 300

model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True,
               input_shape=(maxlen, embedding_dims)))

model.add(Dropout(.2))  # <2>

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

from keras.models import Sequential

from keras.layers import Activation, Dropout, LSTM, Flatten, Dense

from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Dense(64, input_dim=14))

model.add(BatchNormalization())

model.add(Activation('sigmoid'))

model.add(Dense(64, input_dim=14))

model.add(BatchNormalization())

model.add(Activation('sigmoid'))

model.add(Dense(1, activation='sigmoid'))

y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # <1>

y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])  # <2>

true_positives = ((y_pred == y_true) & (y_pred == 1)).sum()

true_positives  # <3>

true_negatives = ((y_pred == y_true) & (y_pred == 0)).sum()

true_negatives  # <4>

false_positives = ((y_pred != y_true) & (y_pred == 1)).sum()

false_positives  # <1>

false_negatives = ((y_pred != y_true) & (y_pred == 0)).sum()

false_negatives  # <2>

confusion = [[true_positives, false_positives],
             [false_negatives, true_negatives]]

confusion

import pandas as pd

confusion = pd.DataFrame(confusion, columns=[1, 0], index=[1, 0])

confusion.index.name = r'pred \ truth'

confusion

precision = true_positives / (true_positives + false_positives)

precision

recall = true_positives / (true_positives + false_negatives)

recall

y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])

rmse = np.sqrt((y_true - y_pred) ** 2) / len(y_true))

rmse

corr = pd.DataFrame([y_true, y_pred]).T.corr()

corr[0][1]

np.mean((y_pred - np.mean(y_pred)) * (y_true - np.mean(y_true))) /
    np.std(y_pred) / np.std(y_true)
