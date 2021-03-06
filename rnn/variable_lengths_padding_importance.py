# Based on: https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
#       and https://www.idiap.ch/~katharas/importance-sampling/ --> Not working for TF 2.2
import numpy as np
import random
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Masking
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback

##### Parameters

n_iterations = 30
n_epochs = 50

eval_method = "loss"
#eval_method = "accuracy"

mask_value = np.finfo(np.float32).min

#####

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True, linewidth=208)
np.random.seed(7)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

dataX = []
dataY = []
for i in range(0, len(alphabet), 1):
    seq_in = alphabet[i:i + 1]
    seq_out = alphabet[(i + 1) % len(alphabet)]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

data = dict()
data["X"] = []
data["y"] = []
for seq_len in range(1, len(alphabet)+1):
    for shift in range(len(alphabet)):
        roll = np.roll(dataX, shift, axis=0)
        X = roll[:seq_len].reshape(seq_len) / float(len(alphabet))
        y = utils.to_categorical(roll[seq_len % len(alphabet)], len(alphabet))
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

class EvaluationCallback(Callback):
    def __init__(self, metric):
        if metric=="loss":
            self.metric = "loss"
        else:
            self.metric = "accuracy"
        print("===> Metric is", metric)

    def on_test_begin(self, logs=None):
        self.score = []

    def on_test_batch_end(self, batch, logs=None):
        if self.metric=="loss":
            self.score.append(logs["loss"])
        else:
            self.score.append(max(1 - logs["accuracy"], 0.01))

X_importance = np.array([1 for x in X])

def sample():
    X_tmp = []
    y_tmp = []
    for i in range(len(X_importance)):
        for _ in range(X_importance[i]):
            X_tmp.append(X[i])
            y_tmp.append(y[i])
    return np.array(X_tmp), np.array(y_tmp)

evalCallback = EvaluationCallback(eval_method)

def calc_importance():
    _, a = model.evaluate(X, y, batch_size=1, callbacks=[evalCallback], verbose=1)
    if a == 1:
        return None
    importance = np.round(evalCallback.score / np.min(evalCallback.score)).astype(int)
    print("Min:", np.min(importance), " Max:", np.max(importance), " Mean:", np.mean(importance), " Median:", np.median(importance), " Percentiles [25, 33, 50, 66, 75, 85, 90, 95]:", np.percentile(importance, [25, 33, 50, 66, 75, 85, 90, 95]))
    print("Score Min:", np.min(evalCallback.score), " Score Max:", np.max(evalCallback.score))
    return importance

start = datetime.datetime.now()
for i in range(n_iterations):
    X_sampled, y_sampled = sample()
    print("Iteration", (i+1), "is training on", X_sampled.shape[0], "samples")
    stat = model.fit(X_sampled, y_sampled, batch_size=64, epochs=n_epochs, shuffle=True, verbose=1)
    X_importance = calc_importance()
    if X_importance is None:
        break
end = datetime.datetime.now()

X = np.unique(X, axis=0)
y_hat = model.predict(X)
for sample in range(X.shape[0]):
    x = [int(round(xx * len(alphabet))) for xx in X[sample].reshape(len(alphabet)) if xx >= 0]
    chars = [int_to_char[round(v)] for v in x]
    index = np.argmax(y_hat[sample])
    print(chars, "=>", int_to_char[index])

print("===> Iterations:", (i+1), "Total epochs:", ((i+1) * n_epochs))
print("===> Total training time:", (end-start))