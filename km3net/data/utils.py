import pandas as pd
import os

def load(path):
    """
    In: path -> Str, path to datafile
    Out: frame representing the processed data. Returns None if
    expectations are not met.
    Expects: `path` to exist.
    """
    if os.path.isfile(path):
        return pd.read_csv(path)
    else:
        print('{0} does not exist.'.format(path))

def get_timeslice_with_hits(data, n=1, index=False, largest=True):
    """
    data -> frame, ideally should be the processed dataset
    In: n-> Int, number of timeslices. Defaults to 1.
    index -> Bool, return only the indices.
    largest -> Bool, return the largest timeslices. Else return the smallest.
    Out: frame representing the n largest timeslice.
    """
    df = data.groupby('timeslice', as_index=False).count()

    if largest:
        df = df.sort_values(by='event_id', ascending=False).head(n)
    else:
        df = df[df['event_id'] < 10]
        df.sort_values(by='event_id', ascending=False).head(n)

    if index:
        return df.index
    else:
        return df

