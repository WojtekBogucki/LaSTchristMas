import pickle
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname, plot_confusion_matrix, visualize
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

def cnn_lstm2(input_dim, output_dim, dropout=0.4, seed=420):
    # Input data type
    reset_random_seeds(seed)
    dtype = 'float32'
    model = Sequential([
        Conv1D(filters=512, kernel_size=10, strides=4, input_shape=input_dim, dtype=dtype),
        Activation('relu'),
        BatchNormalization(),
        Dropout(dropout),
        Conv1D(filters=256, kernel_size=10, strides=4),
        Activation('relu'),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(128, activation='tanh', return_sequences=True, recurrent_activation='sigmoid', dropout=dropout),
        LSTM(128, activation='tanh', return_sequences=False, recurrent_activation='sigmoid', dropout=dropout),
        Dense(units=64, activation='relu'),
        Dropout(dropout),
        Dense(units=output_dim, activation='softmax')
    ])
    return model


input_dim = (99, 161)
n_classes = len(classes)
adam = Adam(lr=1e-4, clipnorm=1.0)
models3 = []
histories3 = []
predictions3 = []
for seed in [420, 1234, 4567]:
    K.clear_session()
    model = cnn_lstm2(input_dim, n_classes, 0.4, seed)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    print("Model  seed: {0}".format(seed))
    history = model.fit(x_train, y_train,
                        batch_size=128, epochs=50,
                        validation_data=(x_val, y_val)
                        )
    pred = model.predict(x_val)
    # plot_confusion_matrix(y_val.argmax(axis=1),pred.argmax(axis=1), normalize=True, classes=classes, filename="model1_drop_{}".format(int(dropout*100)))
    models3.append(model)
    histories3.append(history)
    predictions3.append(pred)


with open("hist2.pickle", "rb") as f:
    pickle.load(f)

labels = list(np.array([[name + " " +str(i) for i in range(1, 4)] for name in ["dropout=0", "dropout=0.2", "dropout=0.4"]]).flatten())
visualize(histories, labels, "loss", title="Comparison of loss on training set", filename="model1_drop")
visualize(histories, labels, "accuracy", title="Comparison of accuracy on training set", filename="model1_drop")
visualize(histories, labels, "val_loss", title="Comparison of loss on validation set", filename="model1_drop")
visualize(histories, labels, "val_accuracy", title="Comparison of accuracy on validation set", filename="model1_drop")

losses=[]
accs=[]
for model in models:
    loss, acc = model.evaluate(x_val, y_val)
    losses.append(loss)
    accs.append(acc)

stats = pd.DataFrame({"model": ["dropout=0", "dropout=0.2", "dropout=0.4"],
                      "avg_loss": [np.mean(losses[:3]),np.mean(losses[3:6]),np.mean(losses[6:9])],
                      "avg_acc": [np.mean(accs[:3]),np.mean(accs[3:6]),np.mean(accs[6:9])]})

stats.to_csv("stats/model1_stats.csv")