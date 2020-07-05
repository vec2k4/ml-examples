# Based on: https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
#       and https://www.idiap.ch/~katharas/importance-sampling/ --> Not working for TF 2.2
import numpy as np
import random
import time
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Masking
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True, linewidth=208)
np.random.seed(7)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet), 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[(i + seq_length) % len(alphabet)]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

mask_value = np.finfo(np.float32).min #-1.0
data = dict()
data["X"] = []
data["y"] = []
for seq_len in range(1, len(alphabet)+1):
    for shift in range(len(alphabet)):
        roll = np.roll(dataX, shift, axis=0)
        X = roll[:seq_len].reshape(seq_len) / float(len(alphabet))
        y = np_utils.to_categorical(roll[seq_len % len(alphabet)], len(alphabet))
        data["X"].append(X)
        data["y"].append(y)

X = pad_sequences(data["X"], maxlen=len(alphabet), value=mask_value, dtype=np.float32)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.array(data["y"])
y = np.reshape(y, (y.shape[0], y.shape[2]))

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(len(alphabet), 1)))
model.add(GRU(32))
model.add(Dense(len(alphabet), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
print(model.summary())

n_iterations = 30
n_epochs = 50

class EvaluationCallback(Callback):
    def on_test_begin(self, logs=None):
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs["loss"])

X_importance = np.array([1 for x in X])

def sample():
    X_tmp = []
    y_tmp = []
    for i in range(len(X_importance)):
        for _ in range(X_importance[i]):
            X_tmp.append(X[i])
            y_tmp.append(y[i])
    return np.array(X_tmp), np.array(y_tmp)

evalCallback = EvaluationCallback()

def calc_importance():
    model.evaluate(X, y, batch_size=1, callbacks=[evalCallback], verbose=1)
    importance = np.round(evalCallback.loss / np.min(evalCallback.loss)).astype(int)
    print("Min: ", np.min(importance), "  Max: ", np.max(importance), "  Mean: ", np.mean(importance), "  Median: ", np.median(importance), "  Percentiles [25, 33, 50, 66, 75, 85, 90, 95]:", np.percentile(importance, [25, 33, 50, 66, 75, 85, 90, 95]))
    print("Loss Min: ", np.min(evalCallback.loss), "  Loss Max: ", np.max(evalCallback.loss))
    return importance

X_sampled, y_sampled = sample()
for i in range(n_iterations):
    print("Iteration: ",i, X_sampled.shape, y_sampled.shape)
    stat = model.fit(X_sampled, y_sampled, batch_size=64, epochs=n_epochs, shuffle=True, verbose=1)
    X_importance = calc_importance()
    X_sampled, y_sampled = sample()

X = np.unique(X, axis=0)
y_hat = model.predict(X)
for sample in range(X.shape[0]):
    x = [int(round(xx * len(alphabet))) for xx in X[sample].reshape(len(alphabet)) if xx >= 0]
    chars = [int_to_char[round(v)] for v in x]
    index = np.argmax(y_hat[sample])
    print(chars, "=>", int_to_char[index])
