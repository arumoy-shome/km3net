import pandas as pd
import numpy as np
from km3net.utils import DATADIR
import os
import re

DATAFILE = DATADIR + "/raw/noise.csv"

def load():
    """
    In: None
    Out: frame representing the raw noise dataset (generated using
    k40gen). None if expectations are not satisfied.
    Expects: 'data/raw/noise.csv' to exist.
    """
    if os.path.isfile(DATAFILE):
        return pd.read_csv(DATAFILE)
    else:
        print('{0} does not exist.'.format(DATAFILE))

def add_positions(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise
    dataset without any position info.
    Expects: data/noise.detx to be a valid file path containing
    position coordinates of noise hits.
    Out: (n, m+3) pandas dataframe with three new columns representing the
    positions.
    """
    df["pos_idx"] = 31 * (noise["dom id"] - 1) + noise["pmt id"]
    pos = read_positions("data/raw/noise.detx")
    pos["pos_idx"] = pos.index
    df = pd.merge(df, pos, on='pos_idx')
    df = df.drop(columns=['pos_idx'])

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

def add_label(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m+1) pandas dataframe with one new column representing the label.
    """
    df["label"] = 0

    return df

def rename_columns(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m) pandas dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'dom id': 'dom_id', 'pmt id': 'pmt_id',
                            'time-over-threshold': 'tot', 'x': 'pos_x', 'y': 'pos_y', 'z':
                            'pos_z', 'dx': 'dir_x', 'dy': 'dir_y', 'dz': 'dir_z'})

    return df

def transform_pmt_id_scheme(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw
    noise dataset where the pmt_id column follows the global numbering
    scheme.
    Out: (n, m) pandas dataframe with pmt_id column now following the
    local numbering scheme.
    """
    df["pmt_id"] = df["pmt_id"] + 1

    return df

def process(write=False):
    """
    In: write -> Bool, write frame to disk as
    'data/processed/noise.csv'. Defaults to False.
    Out: (n, m+4) pandas dataframe with the positions and the label
    columns added, columns renamed and pmt_id changed to the local
    scheme.
    """
    df = load()
    df = add_positions(df)
    df = add_label(df)
    df = rename_columns(df)
    df = transform_pmt_id_scheme(df)

    return df
