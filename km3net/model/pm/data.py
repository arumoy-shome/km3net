import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

class MLPDataset(Dataset):
    def __init__(self, X, y, scale=False):
        """
        In
        --
        X -> (n, m) ndarray, containing the features
        y -> (n,) ndarray, containing the labels
        scale -> Bool, whether to scale data between [0, 1], defaults to False

        Out
        ---
        dataset -> torch.utils.data.Dataset, object
        """
        self.X = X
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        if scale:
            self.scale()

    def __len__(self):
        """
        return number of rows in the dataset
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        return a row at specified id
        """
        return [self.X[idx], self.y[idx]]

    def scale(self):
        self.X = MinMaxScaler().fit_transform(self.X)

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size

        return random_split(self, [train_size, test_size])

def prepare_train_data(path, scale=False, n_test=0.33):
    """
    In
    --
    path -> Str, path to data file.
    scale -> Bool, scale data between [0, 1], defaults to False.

    Expects
    -------
    `path` to be a valid csv file.

    Out
    ---
    train_dl, test_dl -> Tuple, contains the train and test DataLoader iterables
    """
    dataset = MLPDataset(path, scale=scale)
    train, test = dataset.get_splits(n_test)

    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    test_dl = DataLoader(test, batch_size=16, shuffle=True)

    return train_dl, test_dl

def prepare_test_data(path, scale=False):
    """
    In
    --
    path -> Str, path to data file.
    scale -> Bool, scale data between [0, 1], defaults to False.

    Expects
    -------
    `path` to be a valid csv file.

    Out
    ---
    test_dl -> test DataLoader iterable
    """
    dataset = MLPDataset(path, scale=scale)
    _, test = dataset.get_splits(n_test=1.0)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)

    return test_dl
