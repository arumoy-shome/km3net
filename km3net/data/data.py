"""
The dataset for this project comes in two parts. The noise dataset is
generated using the k40gen library. This dataset needs to be combined
with the detx file containing the positions of the noise hits. The
hits dataset comes as a hd5 file with two tables namesly 'mc_hits' and
'mc_info' which need to be combined to obtain some valuable
information. Finally, the two datasets need to be cleaned up and
combined which produces the actual dataset to be used for the project.

This module contains functions to procress the rawdataset into a form
more suitable for further analysis. Although the model exposes several
methods, it is intended to be used by envoking the process method
which returns the final dataset.

Alternatively, one may wish to pick and choose the specific
transformations to create a dataset of their liking.
"""

import pandas as pd
from km3net.data import noise, hits
import os

def add_timeslices(df, duration=15000):
    """
    In: df -> (n, m) pandas dataframe with a time columm. Ideally this
    should be the combined dataset.
    duration -> Int, represents the duration of each timeslice in
    nanoseconds. Defaults to 15000 ns.
    Out: (n, m+1) dataframe with timeslice column added.
    """
    timeslices = list(range(0, math.ceil(df["time"].max())+duration, duration))
    df['timeslice'] = pd.cut(df['time'], bins=timeslices,
    include_lowest=True, labels=False)

    return df

def process(drop=True, sort=True, write=False):
    """
    In: drop -> Bool, drop the dom_id, pmt_id, dir_x, dir_y, dir_z and
    tot columns. Defaults to True.
    sort -> Bool, sort the rows by time. Defaults to True.
    write -> Bool, write frame to disk as 'data/processed/data.csv'. Defaults to False.
    Out: frame with noise and hits dataset combined, rows with
    negative time dropped and timeslice added.
    """
    hits = hits.process()
    noise = noise.process()
    noise["event_id"] = np.nan
    df = pd.concat([hits, noise])
    df = df[df["time"] >= 0.0]
    df = add_timeslices(df)

    if drop:
        df = df.drop(columns=["dom_id", "pmt_id", "dir_x", "dir_y", "dir_z", "tot"])

    if sort:
        df = df.sort_values(by=['time'])

    return df

