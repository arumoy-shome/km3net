import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',
        cmap=plt.cm.Blues):
    """
In
--
cm -> (n, n) Array, the confusion matrix
classes -> List, the classes
normalize -> Bool, normalize the data such that they add up to 100, defaults to False
title -> Str, title of the plot
cmap -> matplotlib.plt.cm, color map to use for the plot
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix with normalization')
    else:
        print('Confusion matrix without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks_marks = np.arange(len(classes))
    plt.xticks(ticks_marks, classes, rotation=45)
    plt.yticks(ticks_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(8,8))

    return plt

def learning_curve(train_losses, valid_losses):
    """
In
--
train_losses -> (n,) List, training losses
valid_losses -> (n,) List or None, validation losses

Out
---
plt -> matplotlib.pyplot, line plot containing the training and validation losses
"""
    plt.title('Learning Curve')
    plt.plot(train_losses)
    plt.plot(valid_losses) if valid_losses
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.figure(figsize=(8,8))

    return plt

def roc_curve(y_true, y_pred):
    pass
