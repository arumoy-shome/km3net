import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from km3net.model.data import CSVDataset

def prepare_data(path):
    """
    In: path -> Str, path to data file.
    Out: Tuple, contains the train and test DataLoader iterables
    Expects: `path` to be a valid csv file.
    """
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    return train_dl, test_dl

def train(data, model, criterion, optimizer, epochs=10):
    """
    In: data -> DataLoader, iterable training data.
    model -> Module, the model to train.
    criterion -> The loss function to use.
    optimizer -> The optimizer to use.
    epochs -> Int, number of epochs to train, defaults to 10.
    Out: None
    """
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(data):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

def test(data, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(data):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        yhat = yhat.round()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)

    return acc
