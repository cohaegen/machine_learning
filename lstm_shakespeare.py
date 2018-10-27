import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import argparse

parse = argparse.ArgumentParser(description='Learn to generate text from an example')
parse.add_argument('--input', type=str, required=True, help='Text input to train on')
parse.add_argument('--sequence_length', type=int, default=50, help='Length of sequences to create from training data')
parse.add_argument('--num_sequences', type=int, help='Number of training sequences to generate')
parse.add_argument('--epochs', type=int, default=10, help='Training epochs')
parse.add_argument('--generate_length', type=int, default=400, help='Number of characters to generate')
parse.add_argument('--model_in', type=str, help='Saved model to load')
parse.add_argument('--model_out', type=str, help='File name to save trained model to')
args = parse.parse_args()

SAMPLE_SEQ_LEN = args.sequence_length
EPOCHS = args.epochs
GENERATE_LEN = args.generate_length


## See if we already have the complete works of William Shakespeare
print('Reading Shakespeare text into memory')
try:
    with open(args.input, 'rb') as file:
        text = file.read()
except:
    print('You must download the Complete Works of William Shakespeare before running this experiment')
    print('You can do so with the following command:')
    print('wget -O shakespeare.txt http://www.gutenberg.org/files/100/100-0.txt')
    raise(RuntimeError('Missing shakespeare.txt'))
# Convert to a list of integers and filter out non-printing characters
text_as_ints = list(filter(lambda c: c <= 0x7f, list(text)))
    

## prepare text for machine learning
print('Preparing text for machine learning')
seq_len = SAMPLE_SEQ_LEN  # number of characters per sample sequence
num_seq = args.num_sequences or int(len(text)/seq_len*1.2)  # number of sequences to sample from the text
def sample_seq(text, seq_len, num_samples):
    samples = []
    indices = np.random.randint(0, len(text) - seq_len - 1, num_samples)
    for i in range(num_samples):
        samples.append(text[indices[i]:(indices[i]+seq_len)])
    return np.array(samples)

training_data = sample_seq(text_as_ints, seq_len+1, num_seq)
x = training_data[:,0:-1]
y = training_data[:,1:]
y_onehot = keras.utils.to_categorical(y, 128)

## Create the Keras DNN model
print('Preparing Keras model')
if(args.model_in):
    print(f'Reading model from {args.model_in}')
    model = keras.models.load_model(args.model_in)
else:
    model = keras.Sequential()
    model.add(keras.layers.Embedding(128, 20))
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(128)))
    model.add(keras.layers.Softmax())
    print('Model output shape: %s' % str(model.output_shape))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[keras.metrics.categorical_accuracy])
print(model.summary())

## Train the model
print('Fitting model')
model.fit(x, y_onehot, epochs=EPOCHS)

if(args.model_out):
    print(f'Saving model to {args.model_out}')
    model.save(args.model_out)

## generate some Shakespeare of our own
def generate_shakespeare(model, string='T', length=100):
    string_as_ints = [ord(c) for c in string]
    for i in range(length):
        # pick the next character based on predicted probabilities
        p = model.predict(np.array([string_as_ints]))[0,-1,:]
        # Dot product with the triangle matrix:
        # This converts p, a series of individual probabilties, to a series of
        # cumulative probabilities
        p_cum = p.dot(np.tri(128).T)
        pick_random = np.where(p_cum > np.random.random())[0]
        if(len(pick_random) < 1):
            # slim chance np.random.random will produce something a little more
            # than the last entry in p_cum (shouldn't happen, but might due to
            # rounding issues)
            next_char = len(pick_random)-1
        else:
            next_char = pick_random[0]
        string_as_ints.append(next_char)
    return ''.join([chr(c) for c in string_as_ints])

print('Generating text')
print(generate_shakespeare(model, string='A', length=GENERATE_LEN))