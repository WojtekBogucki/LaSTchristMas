'''
GRU layers
'''
import pickle
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname, plot_confusion_matrix, visualize2
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


def cnn_gru(input_dim, output_dim, dropout=0.4, seed=420):
    # Input data type
    reset_random_seeds(seed)
    dtype = 'float32'
    model = Sequential([
        Conv1D(filters=256, kernel_size=15, strides=4, input_shape=input_dim, dtype=dtype),
        Activation('relu'),
        BatchNormalization(),
        Dropout(dropout),
        GRU(128, activation='tanh', return_sequences=True, recurrent_activation='sigmoid', dropout=dropout),
        GRU(128, activation='tanh', return_sequences=False, recurrent_activation='sigmoid', dropout=dropout),
        Dense(units=64, activation='relu'),
        Dropout(dropout),
        Dense(units=output_dim, activation='softmax')
    ])
    return model


input_dim = (99, 161)
n_classes = len(classes)
adam = Adam(lr=1e-4, clipnorm=1.0)
models4 = []
histories4 = []
predictions4 = []
for seed in [420, 1234, 4567]:
    K.clear_session()
    model = cnn_gru(input_dim, n_classes, 0.4, seed)
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
    models4.append(model)
    histories4.append(history)
    predictions4.append(pred)


with open("data/conv_layers_test_hist.pickle", "rb") as f:
    histories3 = pickle.load(f)

histories4 = [hist.history for hist in histories4]
histories4 = histories4 + histories3[3:]
with open("data/gru_test_hist.pickle", "wb") as f:
    pickle.dump(histories4, f)

with open("data/gru_test_pred.pickle", "wb") as f:
    pickle.dump(predictions4, f)

labels = list(np.array([[name + " " +str(i) for i in range(1, 4)] for name in ["GRU", "LSTM"]]).flatten())
visualize2(histories4, labels, "loss", title="Comparison of loss on training set")
visualize2(histories4, labels, "accuracy", title="Comparison of accuracy on training set")
visualize2(histories4, labels, "val_loss", title="Comparison of loss on validation set", start_from=30)
visualize2(histories4, labels, "val_accuracy", title="Comparison of accuracy on validation set", start_from=30)

losses4=[]
accs4=[]
for model in models4:
    loss, acc = model.evaluate(x_val, y_val)
    losses4.append(loss)
    accs4.append(acc)

stats3 = pd.read_csv("stats/model1_stats3.csv")
stats4 = pd.DataFrame({"model": ["GRU", "LSTM"],
                     "avg_loss": [np.mean(losses4[:3]), stats3.loc[1, "avg_loss"]],
                      "avg_acc": [np.mean(accs4[:3]),stats3.loc[1, "avg_acc"]]})

stats4.to_csv("stats/model1_stats4.csv")