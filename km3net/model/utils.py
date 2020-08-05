import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from km3net.model.data import CSVDataset

def prepare_train_data(path, normalise=False):
    """
    In: path -> Str, path to data file.
    Out: Tuple, contains the train and test DataLoader iterables
    Expects: `path` to be a valid csv file.
    """
    dataset = CSVDataset(path, normalise=normalise)
    train, test = dataset.get_splits()

    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    test_dl = DataLoader(test, batch_size=16, shuffle=True)

    return train_dl, test_dl

def prepare_test_data(path, normalise=False):
    dataset = CSVDataset(path, normalise=normalise)
    _, test = dataset.get_splits(n_test=1.0)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)

    return test_dl

def train(loader, model, criterion, optimizer):
    """
    In: loader -> DataLoader, iterable training data.
    model -> Module, the model to train.
    criterion -> The loss function to use.
    optimizer -> The optimizer to use.
    epochs -> Int, number of epochs to train, defaults to 10.
    Out: loss_values -> List, loss per epoch
    """
    device = get_device()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = criterion(yhat, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader.dataset)

def valid(loader, model, criterion):
    """
    TODO
    """
    device = get_device()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        loss = criterion(yhat, targets)
        running_loss += loss.item()

    return running_loss / len(loader.dataset)

@torch.no_grad()
def test(loader, model):
    device = get_device()
    all_yhats = torch.tensor([], device=device)
    all_targets = torch.tensor([], device=device)
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        yhat = yhat.round()
        all_targets = torch.cat((all_targets, targets), 0)
        all_yhats = torch.cat((all_yhats, yhat), 0)

    all_targets = all_targets.cpu().numpy()
    all_yhats = all_yhats.cpu().numpy()

    return all_targets, all_yhats

def evaluate(model, optimizer, criterion, epochs, train_dl, valid_dl, test_dl):
    train_losses = []
    valid_losses = []
    print('---')
    for epoch in range(epochs):
        train_loss = train(train_dl, model, criterion, optimizer)
        valid_loss = valid(valid_dl, model, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print statistics every 5 epochs
        if epoch % 5 == 0:
            print("epochs: %d, train loss: %.3f, valid loss: %.3f" % (epoch,
                train_loss, valid_loss))

    print('---')
    y_true, y_pred = test(test_dl, model)

    return {'train_losses': train_losses, 'valid_losses': valid_losses,
            'y_true': y_true, 'y_pred': y_pred}

def get_device():
    """
    In: None
    Out: torch.device, 'cuda' if available else 'cpu'
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device
