import time
import torch

def train(G, model, criterion, optimizer, device):
    """
    In
    --
    G -> torch_geometric.data, training graph.
    model -> Module, the model to train.
    criterion -> The loss function to use.
    optimizer -> The optimizer to use.
    epochs -> Int, number of epochs to train, defaults to 10.

    Out
    ---
    loss -> Float, loss of the epoch
    """
    model.train()
    optimizer.zero_grad()
    G = G.to(device)
    yhat, h = model(G)
    yhat = yhat.squeeze() # (batch_size, 1) -> (batch_size,) to match targets.shape
    loss = criterion(yhat, G.y)
    loss.backward()
    optimizer.step()

    return loss.item(), h

def valid(G, model, criterion, device):
    """
    In
    --
    G -> torch_geometric.data, validation graph.
    model -> Module, the model to train.
    criterion -> The loss function to use.

    Out
    ---
    loss -> Float, loss of the epoch.
    """
    model.eval()
    G = G.to(device)
    yhat, h = model(G)
    yhat = yhat.squeeze() # (batch_size, 1) -> (batch_size,) to match targets.shape
    loss = criterion(yhat, G.y)

    return loss.item(), h

@torch.no_grad()
def test(G, model, device):
    """
    In
    --
    G -> torch_geometric.data, testing graph.
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
    model.eval()
    G = G.to(device)
    yhat, h = model(G)
    yhat = yhat.squeeze()
    yhat_label = yhat.round()
    y_true = torch.cat((y_true, G.y), 0)
    y_pred = torch.cat((y_pred, yhat_label), 0)
    y_score = torch.cat((y_score, yhat), 0)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_score = y_score.cpu().numpy()

    return y_true, y_pred, y_score, h

def evaluate(model, optimizer, criterion, epochs, device, train_G, valid_G, test_G):
    """
    In
    --
    model -> Module, the model to train.
    optimizer -> The optimizer to use.
    criterion -> The loss function to use.
    epochs -> Int, number of epochs to train, defaults to 10.
    train_G -> torch_geometric.data, training graph.
    test_G -> torch_geometric.data, testing graph.
    valid_G -> torch_geometric.data, validation graph.

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
        train_loss, train_h = train(train_G, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        if valid_G:
            valid_loss, valid_h = valid(valid_G, model, criterion, device)
            valid_losses.append(valid_loss)

        # print statistics every epoch
        if valid_G:
            print("epochs: %d, train loss: %.3f, valid loss: %.3f" % (epoch,
                train_loss, valid_loss))
        else:
            print("epochs: %d, train loss: %.3f, valid loss: --" % (epoch,
                train_loss))

    print('---')
    model.eval()
    metrics = []
    for idx, dl in enumerate(test_G):
        y_true, y_pred, y_score, test_h = test(dl, model, device)
        metrics.append({
            'train_losses': train_losses,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'model': model
            })
        metrics[idx]['valid_losses'] = valid_losses if valid_G else None

    return metrics
