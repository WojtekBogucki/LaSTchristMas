'''
A CNN + LSTM model
https://www.kaggle.com/himanshurawlani/a-cnn-lstm-model/notebook
'''
import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
import pandas as pd
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D, GRU
from tensorflow.keras.layers import Lambda, Input, Dropout, Masking, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard

# determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


sample_rate = 16000
all_labels = ['_background_noise_', 'dog', 'four', 'left', 'off', 'seven', 'three', 'wow', 'bed', 'down', 'go',
              'marvin', 'on', 'sheila', 'tree', 'yes', 'bird', 'eight', 'happy', 'nine', 'one', 'six', 'two', 'zero',
              'cat', 'five', 'house', 'no', 'right', 'stop', 'up']
classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

# src folders
# root_path = r'..'
# out_path = r'.'
# model_path = r'.'
train_data_path = os.path.join('data', 'train', 'audio')
# test_data_path = os.path.join(root_path, 'data', 'test', 'audio')

labels, fnames = list_wavs_fname(train_data_path)

new_sample_rate = 16000
y_train = []
x_train = []

for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else:
        n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)

x_train = np.array(x_train)
y_train = label_transform(y_train, classes)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()


def cnn_lstm(input_dim, output_dim, dropout=0.2, n_layers=1):
    #     # Input data type
    reset_random_seeds(420)
    dtype = 'float32'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=input_dim, dtype=dtype)

    # 1 x 1D convolutional layers with strides 4
    x = Conv1D(filters=256, kernel_size=10, strides=4, name='conv_1')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout, name='dropout_1')(x)

    x = LSTM(128, activation='tanh', return_sequences=True, recurrent_activation='sigmoid',
             dropout=dropout, name='lstm_1')(x)
    x = LSTM(128, activation='tanh', return_sequences=False, recurrent_activation='sigmoid',
             dropout=dropout, name='lstm_2')(x)

    #     # 1 fully connected layer DNN ReLu with default 20% dropout
    x = Dense(units=64, activation='relu', name='fc')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=output_dim, activation='softmax', name='softmax')(x)

    network_model = Model(inputs=input_data, outputs=y_pred)

    return network_model


input_dim = (99, 161)
classes = len(classes)
K.clear_session()
model = cnn_lstm(input_dim, classes)
model.summary()

adam = Adam(lr=1e-4, clipnorm=1.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=128, epochs=10,
                    #validation_data=(X_val, Y_val)
                    )

pd.DataFrame(history.history).plot()