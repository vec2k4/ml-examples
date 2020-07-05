# Based on: https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
import numpy as np
import random
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Masking
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences

##### Parameters

n_epochs = 180

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
        if seq_len < 5:
            for i in range(50*(5-seq_len)):
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

start = datetime.datetime.now()
stat = model.fit(X, y, batch_size=64, epochs=n_epochs, shuffle=True, verbose=1)
end = datetime.datetime.now()

X = np.unique(X, axis=0)
y_hat = model.predict(X)
for sample in range(X.shape[0]):
    x = [int(round(xx * len(alphabet))) for xx in X[sample].reshape(len(alphabet)) if xx >= 0]
    chars = [int_to_char[round(v)] for v in x]
    index = np.argmax(y_hat[sample])
    print(chars, "=>", int_to_char[index])

print("===> Total epochs:", n_epochs)
print("===> Total training time:", (end-start))