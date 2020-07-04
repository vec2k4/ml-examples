# Based on: https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, TimeDistributed
from keras.utils import np_utils

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

data = dict()
for seq_len in range(1, len(alphabet)+1):
    data[seq_len] = dict()
    for shift in range(len(alphabet)):
        roll = np.roll(dataX, shift, axis=0)
        data[seq_len][shift] = dict()
        data[seq_len][shift]['X'] = roll[:seq_len] / float(len(alphabet))
        data[seq_len][shift]['y'] = np_utils.to_categorical(roll[seq_len % len(alphabet)], len(alphabet))

model = Sequential()
model.add(GRU(32, return_sequences=False, input_shape=(None, 1)))
model.add(Dense(len(alphabet), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
print(model.summary())

max_iterations = 75
for i in range(max_iterations):
    for seq_len in np.random.permutation(range(1, len(alphabet)+1)):
        shifts = np.random.permutation(len(alphabet))
        X = np.array([data[seq_len][shift]["X"] for shift in shifts])
        y = np.array([data[seq_len][shift]["y"] for shift in shifts]).reshape(len(alphabet), len(alphabet))

        epochs = 10 + int(10 * len(alphabet)/seq_len) + int(10 * (max_iterations - i) / max_iterations)
        stat = model.fit(X, y, batch_size=len(alphabet), epochs=epochs, verbose=0)
        print(f"Iteration: {i+1:3}, SeqLen: {seq_len:2}, Epochs: {epochs:3} => Loss: {stat.history['loss'][-1]:.4f}, Acc: {100*stat.history['accuracy'][-1]:6.2f}%")

    model.reset_states()

    acc = []
    loss = []
    for seq_len in np.random.permutation(range(1, len(alphabet)+1)):
        shifts = np.random.permutation(len(alphabet))
        X = np.array([data[seq_len][shift]["X"] for shift in shifts])
        y = np.array([data[seq_len][shift]["y"] for shift in shifts]).reshape(len(alphabet), len(alphabet))

        l, a = model.evaluate(X, y, batch_size=len(alphabet), verbose=0)
        loss.append(l)
        acc.append(a)

    print(f"===> Model Loss: {np.mean(loss):.4f}, Acc: {100*np.mean(acc):6.2f}%")    


for seq_len in range(1, len(alphabet)+1):
    shifts = range(len(alphabet))
    X = np.array([data[seq_len][shift]["X"] for shift in shifts])
    for x in X:
        y = model.predict(x.reshape(1,seq_len,1), batch_size=len(alphabet))
        index = np.argmax(y)
        x = x.reshape(seq_len) * len(alphabet)
        chars = [int_to_char[round(v)] for v in x]
        print(chars, "=>", int_to_char[index])
