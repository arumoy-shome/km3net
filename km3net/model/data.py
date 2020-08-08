import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split

class CSVDataset(Dataset):
    def __init__(self, path, normalise=False):
        df = pd.read_csv(path, header=None)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
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
