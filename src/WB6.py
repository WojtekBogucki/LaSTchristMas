'''
Bidirectional LSTM
'''
import pickle
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname, plot_confusion_matrix, \
    visualize2
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

def cnn_bilstm3(input_dim, output_dim, dropout=0.4, seed=420, n_layers=2):
    # Input data type
    reset_random_seeds(seed)
    dtype = 'float32'
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=10, strides=4, input_shape=input_dim, dtype=dtype))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    for i in range(n_layers-1):
        model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True, recurrent_activation='sigmoid', dropout=dropout)))
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=False, recurrent_activation='sigmoid', dropout=dropout)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_dim, activation='softmax'))

    return model


input_dim = (99, 161)
n_classes = len(classes)
adam = Adam(lr=1e-4, clipnorm=1.0)

models7 = []
histories7 = []
predictions7 = []

for n_layers in [1, 3]:
    for seed in [420, 1234, 4567]:
        K.clear_session()
        model = cnn_bilstm3(input_dim, n_classes, 0.4, seed, n_layers)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        print("Model  seed: {0} n layers: {1}".format(seed, n_layers))
        history = model.fit(x_train, y_train,
                            batch_size=128, epochs=50,
                            validation_data=(x_val, y_val)
                            )
        pred = model.predict(x_val)
        # plot_confusion_matrix(y_val.argmax(axis=1),pred.argmax(axis=1), normalize=True, classes=classes, filename="model1_drop_{}".format(int(dropout*100)))
        models7.append(model)
        histories7.append(history)
        predictions7.append(pred)


with open("data/bilstm_size_test_hist.pickle", "rb") as f:
    histories3 = pickle.load(f)

histories7 = [hist.history for hist in histories7]
histories7= histories7 + histories3[6:]
with open("data/bilstm_n_layers_test_hist.pickle", "wb") as f:
    pickle.dump(histories7, f)

with open("data/bilstm_n_layers_test_pred.pickle", "wb") as f:
    pickle.dump(predictions7, f)

labels = list(np.array([[name + " " +str(i) for i in range(1, 4)] for name in ["1 bilstm layer", "3 bilstm layers", "2 bilstm layers"]]).flatten())
visualize2(histories7, labels, "loss", title="Comparison of loss on training set")
visualize2(histories7, labels, "accuracy", title="Comparison of accuracy on training set")
visualize2(histories7, labels, "val_loss", title="Comparison of loss on validation set", start_from=10)
visualize2(histories7, labels, "val_accuracy", title="Comparison of accuracy on validation set", start_from=20)

losses7=[]
accs7=[]
for model in models7:
    loss, acc = model.evaluate(x_val, y_val)
    losses7.append(loss)
    accs7.append(acc)

stats6 = pd.read_csv("stats/model1_stats6.csv")
stats7 = pd.DataFrame({"model": ["1 bilstm layer", "3 bilstm layers", "2 bilstm layers"],
                       "avg_loss": [np.mean(losses7[:3]), np.mean(losses7[3:]), stats6.loc[2, "avg_loss"]],
                       "avg_acc": [np.mean(accs7[:3]),np.mean(accs7[3:]),stats6.loc[2, "avg_acc"]]})

stats7.to_csv("stats/model1_stats7.csv")