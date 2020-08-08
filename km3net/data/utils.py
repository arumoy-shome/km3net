import pandas as pd
import os

def get_timeslice_with_hits(data, n=1, index=False, largest=True):
    """
    In
    --
    n-> Int, number of timeslices. Defaults to 1.
    data -> (n, m) frame, ideally should be the processed dataset
    index -> Bool, return only the indices.
    largest -> Bool, return the largest timeslices. Else return the smallest.

    Out:
    ---
    df -> (l, k) frame representing the n largest/smallest timeslices or
    indices -> (l,) series representing the indices
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

