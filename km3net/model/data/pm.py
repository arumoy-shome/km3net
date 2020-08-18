import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

class MLPDataset(Dataset):
    def __init__(self, path, normalise=False):
        df = pd.read_csv(path, header=None)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        if normalise:
            self.normalise()

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

    def normalise(self):
        self.X = MinMaxScaler().fit_transform(self.X)

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size

        return random_split(self, [train_size, test_size])

def prepare_train_data(path, normalise=False, n_test=0.33):
    """
    In
    --
    path -> Str, path to data file.
    normalise -> Bool, normalise data between [0, 1], defaults to False.

    Expects
    -------
    `path` to be a valid csv file.

    Out
    ---
    train_dl, test_dl -> Tuple, contains the train and test DataLoader iterables
    """
    dataset = MLPDataset(path, normalise=normalise)
    train, test = dataset.get_splits(n_test)

    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    test_dl = DataLoader(test, batch_size=16, shuffle=True)

    return train_dl, test_dl

def prepare_test_data(path, normalise=False):
    """
    In
    --
    path -> Str, path to data file.
    normalise -> Bool, normalise data between [0, 1], defaults to False.

    Expects
    -------
    `path` to be a valid csv file.

    Out
    ---
    test_dl -> test DataLoader iterable
    """
    dataset = MLPDataset(path, normalise=normalise)
    _, test = dataset.get_splits(n_test=1.0)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)

    return test_dl
