import numpy as np
import torch
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
    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)

    return train_dl, test_dl

def train(loader, model, criterion, optimizer, epochs=10):
    """
    In: loader -> DataLoader, iterable training data.
    model -> Module, the model to train.
    criterion -> The loss function to use.
    optimizer -> The optimizer to use.
    epochs -> Int, number of epochs to train, defaults to 10.
    Out: None
    """
    device = get_device()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 2000))
                running_loss = 0.0

@torch.no_grad()
def test(loader, model):
    device = get_device()
    correct = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        correct += (yhat.round() == targets).float().sum() # per minibatch
        # print stats
        if i % 2000 == 1999:
            print('[{0}, {1}] correct: {2}'.format(1, i+1, correct))

    return correct / len(loader.dataset)

def get_device():
    """
    In: None
    Out: torch.device, 'cuda' if available else 'cpu'
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device
