import numpy as np
from scipy import signal
import pandas as pd
from glob import glob
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model, Sequential


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def pad_audio(samples, sample_rate=16000):
    if len(samples) >= sample_rate:
        return samples
    else:
        return np.pad(samples, pad_width=(sample_rate - len(samples), 0), mode='constant', constant_values=(0, 0))


def chop_audio(samples, sample_rate=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - sample_rate)
        yield samples[beg: beg + sample_rate]


def label_transform(labels, classes):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in classes:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))


def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+\\(\w+)\\\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+\\(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if filename:
        plt.savefig("plots/{0}".format(filename))
        plt.close(fig)
    return ax


def visualize(histories, labels=None, type="val_accuracy", filename=None, start_from=0, title=None):
    fig = plt.figure()
    if not title:
        title = type
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(histories) + 1)]
    for hist, label in zip(histories, labels):
        y = np.array(hist.history[type])
        plt.plot(range(start_from + 1, len(y) + 1), y[start_from:], label=label)
        plt.title(title)
        plt.xlabel("number of epoch")
        plt.ylabel(type)
        plt.legend()
    if filename:
        plt.savefig("plots/{0}_{1}".format(filename, type))
        plt.close(fig)
    else:
        plt.show()

def visualize2(histories, labels=None, type="val_accuracy", filename=None, start_from=0, title=None):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cm = plt.get_cmap('gist_rainbow')
    # NUM_COLORS = len(histories)
    # ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    if not title:
        title = type
    if not labels:
        labels = ["Model {0}".format(j) for j in range(1, len(histories) + 1)]
    for hist, label in zip(histories, labels):
        y = np.array(hist[type])
        plt.plot(range(start_from + 1, len(y) + 1), y[start_from:], label=label)
        plt.title(title)
        plt.xlabel("number of epoch")
        plt.ylabel(type)
        plt.legend()
    if filename:
        plt.savefig("plots/{0}_{1}".format(filename, type))
        plt.close(fig)
    else:
        plt.show()

# https://github.com/yungshun317/keras-rnn-speech-recognizer/blob/master/train_utils.py
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
        outputs=loss_out)
    return model