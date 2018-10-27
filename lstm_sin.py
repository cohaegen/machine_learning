import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

x = np.linspace(0, 10*np.pi, num=100000)
y = np.sin(x)
train_x = x[0:80000]
train_y = y[0:80000]
test_x = x[80000:]
test_y = y[80000:]

## Generate training data sequences
def generate_sample(x, seq_len):
    '''Return a sample sequence of data from x that is seq_len long'''
    seq_start = np.random.randint(0, len(x) - seq_len - 1)
    return x[seq_start:(seq_start+seq_len)]

train_seqs_x = []
train_seqs_y = []
for i in range(100000):
    seq = generate_sample(train_y, 51)
    #train_seq_y = seq[1:]
    train_seq_y = seq[-1]
    train_seq_x = seq[0:-1]
    train_seqs_x.append(train_seq_x)
    train_seqs_y.append(train_seq_y)
train_seqs_x = np.array(train_seqs_x).reshape((-1, 50, 1))
train_seqs_y = np.array(train_seqs_y)
#train_seqs_y = np.array(train_seqs_y).reshape((-1, 50, 1))

## Generate test data sequences
test_seqs_x = []
test_seqs_y = []
for i in range(0,len(test_x)-50 - 1):
    test_seqs_x.append(test_y[i:(i+50)])
    test_seqs_y.append(test_y[i+50])
test_seqs_x = np.array(test_seqs_x).reshape(len(test_seqs_x),-1,1)
test_seqs_y = np.array(test_seqs_y)

## Create the model
model = keras.Sequential()
model.add(keras.layers.LSTM(20, input_shape=(50,1), return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse')

## Fit the model
model.fit(train_seqs_x, train_seqs_y, epochs=10)
#y = [model.predict(test_seqs_x[i].reshape(1,50,1)) for i in range(len(test_seqs_x))]
#y = np.array(y)
#y = y.reshape(-1,)
y = model.predict(test_seqs_x)
plt.plot(y)
plt.show()