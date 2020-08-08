import pandas as pd
from km3net.data import noise, hits
import os

def add_timeslices(df, duration=15000):
    """
    In
    --
    df -> (n, m) dataframe with a time columm. Ideally this should be the
    combined dataset.
    duration -> Int, represents the duration of each timeslice in
    nanoseconds. Defaults to 15000 ns.

    Out
    ---
    df -> (n, m+1) dataframe with timeslice column added.
    """
    timeslices = list(range(0, math.ceil(df["time"].max())+duration, duration))
    df['timeslice'] = pd.cut(df['time'], bins=timeslices,
    include_lowest=True, labels=False)

    return df

def process(drop=True, sort=True):
    """
    In
    --
    drop -> Bool, drop the dom_id, pmt_id, dir_x, dir_y, dir_z and tot
    columns. Defaults to True.
    sort -> Bool, sort the rows by time. Defaults to True.

    Out
    ---
    frame with noise and hits dataset combined, rows with negative time
    dropped and timeslice added.
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

