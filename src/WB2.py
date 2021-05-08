import pickle
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname, plot_confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D, GRU
from tensorflow.keras.layers import Lambda, Input, Dropout, Masking, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard

# determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


with open("data/x_train.pickle", "rb") as f:
    x_train = pickle.load(f)
with open("data/y_train.pickle", "rb") as f:
    y_train = pickle.load(f)
with open("data/x_val.pickle", "rb") as f:
    x_val = pickle.load(f)
with open("data/y_val.pickle", "rb") as f:
    y_val = pickle.load(f)

classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

def cnn_lstm2(input_dim, output_dim, dropout=0.2, n_layers=1):
    # Input data type
    reset_random_seeds(420)
    dtype = 'float32'
    model = Sequential([
        Conv1D(filters=512, kernel_size=15, strides=4, input_shape=input_dim, dtype=dtype),
        Activation('relu'),
        BatchNormalization(),
        Dropout(dropout),
        Conv1D(filters=256, kernel_size=15, strides=4),
        Activation('relu'),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(512, activation='tanh', return_sequences=True, recurrent_activation='sigmoid', dropout=dropout),
        LSTM(512, activation='tanh', return_sequences=False, recurrent_activation='sigmoid', dropout=dropout),
        Dense(units=128, activation='relu'),
        Dropout(dropout),
        Dense(units=output_dim, activation='softmax')
    ])
    return model


input_dim = (99, 161)
n_classes = len(classes)
K.clear_session()
model2 = cnn_lstm2(input_dim, n_classes)
model2.summary()

adam = Adam(lr=1e-4, clipnorm=1.0)

model2.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
history2 = model2.fit(x_train, y_train,
                    batch_size=128, epochs=20,
                    validation_data=(x_val, y_val)
                    )

pd.DataFrame(history2.history).plot()
model2.evaluate(x_val, y_val)
pred2 = model2.predict(x_val)

plot_confusion_matrix(y_val.argmax(axis=1),pred2.argmax(axis=1), normalize=True, classes=classes, filename="model2_conf_mat")