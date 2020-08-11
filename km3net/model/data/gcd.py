import pandas as pd
import torch
import dgl
import km3net.data.pm as pm
import km3net.data.gcd as gcd
from dgl import DGLGraph
from torch.utils.data import Dataset, DataLoader

class GNNDataset(Dataset):
    def __init__(self, path, n=100, size=20):
        df = pd.read_csv(path)
        self.samples = self.generate(n, size, df)

    def __len__(self):
        """
        return number of rows in the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        return a row at specified id
        """
        return self.samples[idx]

    def generate(self, n, size, df):
        samples = []

        for _ in range(n):
            sample = df.sample(size)
            sample = pm.process(sample)
            matrix, label = gcd.process(sample['label'].values, olen=size)
            matrix, label = matrix.astype('float32'), label.astype('float32')
            samples.append((DGLGraph(matrix), label))

        return samples

    def get_splits(self, n_test=0.2):
        test_size = round(n_test * len(self.samples))
        train_size = len(self.samples) - test_size

        return self.samples[:train_size], self.samples[train_size:]

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels)

def prepare_data(path, n=100, size=20):
    """
    In
    --
    path -> Str, path to data file.
    n -> Int, number of graphs
    size -> Int, min number of nodes in each graph

    Expects
    -------
    `path` to be a valid csv file.

    Out
    ---
    train_dl, test_dl -> Tuple, contains the train and test DataLoader iterables
    """
    dataset = GNNDataset(path, n, size)
    train, test = dataset.get_splits()

    train_dl = DataLoader(train, batch_size=16, shuffle=True,
            collate_fn=collate)
    test_dl = DataLoader(test, batch_size=32, shuffle=False,
            collate_fn=collate)

    return train_dl, test_dl

