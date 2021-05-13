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
import pickle
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
import random
import tensorflow as tf
from src.utils import log_specgram, pad_audio, chop_audio, label_transform, list_wavs_fname, plot_confusion_matrix, visualize
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
all_files = set([os.path.join(label, fname) for label, fname in zip(labels, fnames)])

with open("data/train/validation_list.txt", "r") as f:
    validation_files = f.read().splitlines()
validation_files = set([os.path.normpath(v) for v in validation_files])
train_files = list(all_files.difference(validation_files))
validation_files = list(validation_files)
train_files = train_files + [v for v in validation_files if v.startswith("_background")]
new_sample_rate = 16000
y_train = []
x_train = []
y_val = []
x_val = []
reset_random_seeds(420)
for i, train_file in enumerate(train_files):
    if not i%1000: print(i)
    label, _ = train_file.split("\\")
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, train_file))
    samples = pad_audio(samples)
    if len(samples) > 16000 and label == '_background_noise_':
        n_samples = chop_audio(samples, sample_rate=sample_rate, num=400)
    elif len(samples) > 16000 and not label == '_background_noise_':
        n_samples = chop_audio(samples, sample_rate=sample_rate, num=1)
    else:
        n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)

for validation_file in validation_files:
    label, _ = validation_file.split("\\")
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, validation_file))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples, sample_rate=sample_rate, num=50)
    else:
        n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_val.append(label)
        x_val.append(specgram)


x_train = np.array(x_train)
y_train = label_transform(y_train, classes)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)

x_val = np.array(x_val)
y_val = label_transform(y_val, classes)
y_val = y_val.values
y_val = np.array(y_val)

del labels, fnames
gc.collect()

with open("data/x_train.pickle", "wb") as f:
    pickle.dump(x_train, f)
with open("data/y_train.pickle", "wb") as f:
    pickle.dump(y_train, f)
with open("data/x_val.pickle", "wb") as f:
    pickle.dump(x_val, f)
with open("data/y_val.pickle", "wb") as f:
    pickle.dump(y_val, f)


with open("data/x_train.pickle", "rb") as f:
    x_train = pickle.load(f)
with open("data/y_train.pickle", "rb") as f:
    y_train = pickle.load(f)
with open("data/x_val.pickle", "rb") as f:
    x_val = pickle.load(f)
with open("data/y_val.pickle", "rb") as f:
    y_val = pickle.load(f)


def cnn_lstm(input_dim, output_dim, dropout=0.2, seed=420, kernel_size=10):
    # Input data type
    reset_random_seeds(seed)
    dtype = 'float32'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=input_dim, dtype=dtype)

    # 1 x 1D convolutional layers with strides 4
    x = Conv1D(filters=256, kernel_size=kernel_size, strides=4, name='conv_1')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout, name='dropout_1')(x)

    x = LSTM(128, activation='tanh', return_sequences=True, recurrent_activation='sigmoid',
             dropout=dropout, name='lstm_1')(x)
    x = LSTM(128, activation='tanh', return_sequences=False, recurrent_activation='sigmoid',
             dropout=dropout, name='lstm_2')(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = Dense(units=64, activation='relu', name='fc')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=output_dim, activation='softmax', name='softmax')(x)

    network_model = Model(inputs=input_data, outputs=y_pred)

    return network_model


input_dim = (99, 161)
n_classes = len(classes)
adam = Adam(lr=1e-4, clipnorm=1.0)
models = []
histories = []
predictions = []
for dropout in [0, 0.2, 0.4]:
    for seed in [420, 1234, 4567]:
        K.clear_session()
        model = cnn_lstm(input_dim, n_classes, dropout, seed)

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        print("Model dropout: {0}, seed: {1}".format(dropout, seed))
        history = model.fit(x_train, y_train,
                            batch_size=128, epochs=50,
                            validation_data=(x_val, y_val)
                            )

        pred = model.predict(x_val)
        # plot_confusion_matrix(y_val.argmax(axis=1),pred.argmax(axis=1), normalize=True, classes=classes, filename="model1_drop_{}".format(int(dropout*100)))
        models.append(model)
        histories.append(history)
        predictions.append(pred)


with open("hist1.pickle", "wb") as f:
    pickle.dump([hist.history for hist in histories], f)
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
#########
models2 = []
histories2 = []
predictions2 = []
for kernel_size in [5, 15]:
    for seed in [420, 1234, 4567]:
        K.clear_session()
        model = cnn_lstm(input_dim, n_classes, 0.4, seed, kernel_size)

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        print("Model kernel size: {0}, seed: {1}".format(kernel_size, seed))
        history = model.fit(x_train, y_train,
                            batch_size=128, epochs=50,
                            validation_data=(x_val, y_val)
                            )

        pred = model.predict(x_val)
        # plot_confusion_matrix(y_val.argmax(axis=1),pred.argmax(axis=1), normalize=True, classes=classes, filename="model1_drop_{}".format(int(dropout*100)))
        models2.append(model)
        histories2.append(history)
        predictions2.append(pred)

histories2 = histories2 + histories[6:]
with open("hist2.pickle", "wb") as f:
    pickle.dump([hist.history for hist in histories2], f)
labels2 = list(np.array([[name + " " +str(i) for i in range(1, 4)] for name in ["kernel_size=5", "kernel_size=15", "kernel_size=10"]]).flatten())
visualize(histories2, labels2, "loss", title="Comparison of loss on training set", filename="model1_ker_size")
visualize(histories2, labels2, "accuracy", title="Comparison of accuracy on training set", filename="model1_ker_size")
visualize(histories2, labels2, "val_loss", title="Comparison of loss on validation set", filename="model1_ker_size")
visualize(histories2, labels2, "val_accuracy", title="Comparison of accuracy on validation set", filename="model1_ker_size")

losses2=[]
accs2=[]
for model in models2:
    loss, acc = model.evaluate(x_val, y_val)
    losses2.append(loss)
    accs2.append(acc)

stats2 = pd.DataFrame({"model": ["kernel_size=5", "kernel_size=15", "kernel_size=10"],
                      "avg_loss": [np.mean(losses2[:3]),np.mean(losses2[3:6]),np.mean(losses[6:9])],
                      "avg_acc": [np.mean(accs2[:3]),np.mean(accs2[3:6]),np.mean(accs[6:9])]})

stats2.to_csv("stats/model1_stats2.csv")