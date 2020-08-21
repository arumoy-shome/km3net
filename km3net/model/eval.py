import itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

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

    if valid_losses:
        plt.plot(valid_losses)

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.figure(figsize=(8,8))

    return plt

def roc_curve(y_true, y_score):
    """
    In
    --
    y_true -> (n,) List, the true labels
    y_score -> (n,) List, probability estimates of the positive class

    Out
    ---
    plt -> matplotlib.pyplot, line plot containing the ROC curve
    """
    ns_score = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = skmetrics.roc_curve(y_true, ns_score)
    fpr, tpr, _ = skmetrics.roc_curve(y_true, y_score)

    plt.title('ROC Curve')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.figure(figsize=(8,8))

    auc = skmetrics.roc_auc_score(y_true, y_score)

    return auc, plt

def precision_recall_curve(y_true, y_score):
    """
    In
    --
    y_true -> (n,) List, the true labels
    y_score -> (n,) List, probability estimates of the positive class

    Out
    ---
    plt -> matplotlib.pyplot, line plot precision-recall curve
    """
    precision, recall, _ = skmetrics.precision_recall_curve(y_true, y_score)
    no_skill = len(y_true[y_true==1]) / len(y_true)

    plt.title('Precision Recall Curve')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.figure(figsize=(8,8))

    auc = skmetrics.auc(recall, precision)

    return auc, plt

def evaluate(y_true, y_pred, y_score, train_losses, valid_losses, model):
    """
    In
    --
    y_true -> (n,) List, true labels
    y_pred -> (n,) List, predicted labels
    y_score -> (n,) List, probability estimates of positive class
    train_losses -> (n,) List, training losses
    valid_losses -> (n,) List or None, validation losses
    model -> torch model, not used, kept for convenience of call this method
    using kwargs

    Out
    ---
    None
    """
    # learning curve
    fig = learning_curve(train_losses, valid_losses)
    fig.show()

    # confusion matrix
    matrix = skmetrics.confusion_matrix(y_true, y_pred)
    fig = confusion_matrix(matrix, classes=[0, 1], normalize=True)
    fig.show()

    # roc curve
    roc_auc, fig = roc_curve(y_true, y_score)
    fig.show()

    # precision recall curve
    pr_auc, fig = precision_recall_curve(y_true, y_score)
    fig.show()

    print('Classification report:')
    print(skmetrics.classification_report(y_true, y_pred))
    print('ROC AUC: %.3f' % roc_auc)
    print('Precision Recall AUC: %.3f' % pr_auc)
    print('F1 Score: %.3f' % skmetrics.f1_score(y_true, y_pred))
    print('F2 Score: %.3f' % skmetrics.fbeta_score(y_true, y_pred, beta=2.0))

