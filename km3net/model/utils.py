import torch

def train(loader, model, criterion, optimizer, device):
    """
    In
    --
    loader -> DataLoader, iterable training data.
    model -> Module, the model to train.
    criterion -> The loss function to use.
    optimizer -> The optimizer to use.
    epochs -> Int, number of epochs to train, defaults to 10.

    Out
    ---
    running_loss -> Float, loss across all mini batches ie. for the entire
    epoch
    """
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        yhat = model(inputs)
        yhat = yhat.squeeze() # (batch_size, 1) -> (batch_size,) to match targets.shape
        loss = criterion(yhat, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader.dataset)

def valid(loader, model, criterion, device):
    """
    In
    --
    loader -> DataLoader, iterable training data.
    model -> Module, the model to train.
    criterion -> The loss function to use.

    Out
    ---
    running_loss -> Float, loss across all mini batches ie. for the entire
    epoch
    """
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        yhat = yhat.squeeze()
        loss = criterion(yhat, targets)
        running_loss += loss.item()

    return running_loss / len(loader.dataset)

@torch.no_grad()
def test(loader, model, device):
    """
    In
    --
    loader -> DataLoader, iterable training data.
    model -> Module, the model to train.

    Expects
    -------
    model to return probability estimates ie. the last activation
    function of model should be a Sigmoid

    Out
    ---
    y_true, y_pred, y_score -> Tuple, containing the true
    targets, predicted targets and probability estimate of positive
    class
    """
    y_pred = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    y_true = torch.tensor([], device=device)
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        yhat = yhat.squeeze()
        yhat_label = yhat.round()
        y_true = torch.cat((y_true, targets), 0)
        y_pred = torch.cat((y_pred, yhat_label), 0)
        y_score = torch.cat((y_score, yhat), 0)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_score = y_score.cpu().numpy()

    return y_true, y_pred, y_score

def evaluate(model, optimizer, criterion, epochs, device, train_dl, valid_dl, test_dl):
    """
    In
    --
    model -> Module, the model to train.
    optimizer -> The optimizer to use.
    criterion -> The loss function to use.
    epochs -> Int, number of epochs to train, defaults to 10.
    train_dl -> DataLoader, training set
    valid_dl -> DataLoader, validation set
    test_dl -> List, containing test set(s)

    Out
    ---
    metrics -> Dict, containing the training and validation losses for
    all epochs, predicted labels, true labels, probability estimate
    of positive class and the model after training
    """
    train_losses = []
    valid_losses = []
    model.train()
    print('---')
    for epoch in range(epochs):
        train_loss = train(train_dl, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        if valid_dl:
            valid_loss = valid(valid_dl, model, criterion, device)
            valid_losses.append(valid_loss)

        # print statistics every epoch
        if valid_dl:
            print("epochs: %d, train loss: %.3f, valid loss: %.3f" % (epoch,
                train_loss, valid_loss))
        else:
            print("epochs: %d, train loss: %.3f, valid loss: --" % (epoch,
                train_loss))

    print('---')
    model.eval()
    metrics = []
    for idx, dl in enumerate(test_dl):
        y_true, y_pred, y_score = test(dl, model, device)
        metrics.append({
            'train_losses': train_losses,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'model': model
            })
        metrics[idx]['valid_losses'] = valid_losses if valid_dl else None

    return metrics

def get_device():
    """
    In
    --
    None

    Out
    ---
    device -> Str, 'cuda' if available else 'cpu'
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device

def save_model(model, path):
    """
In
--
model -> torch.nn, model to save
path -> Str, path to save state_dict of model

Expects
-------
path to be valid

Out
---
None
"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
In
--
model -> torch.nn, instance of model class to load
path -> Str, path where model's state_dict is saved

Expects
-------
path to be valid

Out
---
model -> torch.nn, loaded pytorch model in eval mode
"""
    model = model
    model.load_state_dict(torch.load(path))
    model.eval()

    return model
