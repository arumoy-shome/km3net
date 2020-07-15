import pandas as pd
from data import data

def get_timeslice_with_hits(n=1, index=False, largest=True, data):
    """
    In: n-> Int, number of timeslices. Defaults to 1.
    index -> Bool, return only the indices.
    largest -> Bool, return the largest timeslices. Else return the smallest.
    data -> frame, ideally should be the processed dataset
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

