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
methods, it is intended to be used by envoking the combine method
which accepts the noise and hits datasets as frames and returns the
final dataset.

Alternatively, one may wish to pick and choose the specific
transformations to create a dataset of their liking.
"""

import pandas as pd
import numpy as np
import re

def add_noise_pos(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise
    dataset without any position info.
    Expects: data/noise.detx to be a valid file path containing
    position coordinates of noise hits.
    Out: (n, m+3) pandas dataframe with three new columns representing the
    positions.
    """
    df["pos_idx"] = 31 * (noise["dom id"] - 1) + noise["pmt id"]
    pos = read_positions("data/noise.detx")
    pos["pos_idx"] = pos.index
    df = pd.merge(df, pos, on='pos_idx')
    df = df.drop(columns=['pos_idx'])

    return df

 def add_noise_label(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m+1) pandas dataframe with one new column representing the label.
    """
    df["label"] = 0

    return df

def add_hits_label(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw hits dataset.
    Out: (n, m+1) pandas dataframe with one new column representing the label.
    """
    df["label"] = 1

    return df

def add_hits_event_id(hits):
    """
    In: hits -> (n, m) pandas dataframe, this should be the raw hits dataset (mc_hits).
    Expects: data/events.h5 to be a valid file path.
    Out: (n, m+1) hits with their corresponding event id added as a new column.
    """
    info = pd.read_hdf("data/events.h5", key="/data/mc_info")
    info["event_id"] = info.index
    hits["id"] = hits.index
    info = info.drop_duplicates(subset='nu.hits.end')
    bins = pd.concat([pd.Series([0]), info["nu.hits.end"]])
    hits["event_id"] = pd.cut(hits.id, bins=bins, right=False,
    labels=info["event_id"], include_lowest=True)
    hits = hits.drop(columns=['id'])

    return hits

def rename_noise_col(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m) pandas dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'dom id': 'dom_id', 'pmt id': 'pmt_id',
    'time-over-threshold': 'tot', 'x': 'pos_x', 'y': 'pos_y', 'z':
    'pos_z', 'dx': 'dir_x', 'dy': 'dir_y', 'dz': 'dir_z'})

    return df

def rename_hits_col(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw hits dataset.
    Out: (n, m) pandas dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'h.dom_id': 'dom_id', 'h.pmt_id':
    'pmt_id', 'h.pos.x': 'pos_x', 'h.pos.y': 'pos_y', 'h.pos.z':
    'pos_z', 'h.dir.x': 'dir_x', 'h.dir.y': 'dir_y', 'h.dir.z':
    'dir_z', 'h.tot': 'tot', 'h.t': 'time'})

    return df

def convert_noise_pmt_id(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw
    noise dataset where the pmt_id column follows the global numbering
    scheme.
    Out: (n, m) pandas dataframe with pmt_id column now following the
    local numbering scheme.
    """
    df["pmt_id"] = df["pmt_id"] + 1

    return df

def convert_hits_pmt_id(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw
    hits dataset where the pmt_id column follows the global numbering
    scheme.
    Out: (n, m) pandas dataframe with pmt_id column now following the
    local numbering scheme.
    """
    df["pmt_id"] = df["pmt_id"] - 31 * (df["dom_id"] - 1)

    return df

def read_positions(filename, pmt2index=lambda pmt: pmt - 1):
    """
    In: filename -> path to a detx file containing the positions of noise hits.
    pmt2index -> function to convert pmt ids to indices, set to a reasonable default.
    Out: (n, 3) numpy array.
    """
    line_expr = re.compile(r"\s*(\d+)\s+(-?\d+\.\d+\s*){7}")
    float_expr = re.compile(r"-?\d+\.\d+")

    position_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')])
    positions = np.zeros(115 * 18 * 31, dtype=position_dt)
    with open(filename) as det_file:
        for line in det_file:
            line = line.strip()
            m = line_expr.match(line)
            if m:
                idx = pmt2index(int(m.group(1)))
                positions[idx] = tuple(float(e) for e in float_expr.findall(line[m.end(1):])[:6])

    return positions

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

def process_noise(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m+4) pandas dataframe with the positions and the label
    columns added, columns renamed and pmt_id changed to the local
    scheme.
    """
    df = add_noise_pos(df)
    df = add_noise_label(df)
    df = rename_noise_col(df)
    df = convert_noise_pmt_id(df)

    return df

def process_hits(df):
    """
    In: df -> pandas dataframe, this should be the raw hits dataset (mc_hits).
    Out: (n, m+2) pandas dataframe with the label and event_id columns
    added, columns renamed and pmt_id changed to the local scheme.
    """
    df = rename_hits_col(hits)
    df = add_hits_label(hits)
    df = add_hits_event_id(hits)
    df = convert_hits_pmt_id(hits)

    return df


def combine(hits, noise, drop=True, sort=True):
    """
    In: hits -> (n, m) pandas dataframe, the raw hits dataset (mc_hits).
    noise -> (p, q) pandas dataframe, the raw noise dataset generated from k40gen.
    drop -> Bool, drop the dom_id, pmt_id, dir_x, dir_y, dir_z and tot columns. Defaults to True.
    sort -> Bool, sort the rows by time. Defaults to True.
    """
    hits = process_hits(hits)
    noise = process_noise(noise)
    noise["event_id"] = np.nan
    df = pd.concat([hits, noise])
    df = df[df["time"] >= 0.0]
    df = add_timeslices(df)

    if drop:
        df = df.drop(columns=["dom_id", "pmt_id", "dir_x", "dir_y", "dir_z", "tot"])

    if sort:
        df = df.sort_values(by=['time'])

    return df
