import pandas as pd
import numpy as np
import os
import re

def add_positions(df):
    """
    Input
    -----
    df -> (n, m) dataframe, this should ideally be the raw noise dataset
    without any position info.

    Expects
    -------
    data/raw/noise.detx to be a valid file path containing position
    coordinates of noise hits.

    Out
    ---
    df -> (n, m+3) dataframe with three new columns representing the
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
    In
    --
    filename -> path to a detx file containing the positions of noise hits.
    pmt2index -> function to convert pmt ids to indices, set to a reasonable default.

    Out
    ---
    positions -> (n, 3) numpy array.
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
    In
    --
    df -> (n, m) dataframe, this should ideally be the raw noise dataset.

    Out
    ---
    df -> (n, m+1) dataframe with one new column representing the label.
    """
    df["label"] = 0

    return df

def rename_columns(df):
    """
    In
    --
    df -> (n, m) dataframe, this should ideally be the raw noise dataset.

    Out
    --
    df -> (n, m) dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'dom id': 'dom_id', 'pmt id': 'pmt_id',
                            'time-over-threshold': 'tot', 'x': 'pos_x', 'y': 'pos_y', 'z':
                            'pos_z', 'dx': 'dir_x', 'dy': 'dir_y', 'dz': 'dir_z'})

    return df

def transform_pmt_id_scheme(df):
    """
    In
    --
    df -> (n, m) dataframe, this should ideally be the raw noise dataset where
    the pmt_id column follows the global numbering scheme.

    Out
    ---
    df -> (n, m) dataframe with pmt_id column now following the local
    numbering scheme.
    """
    df["pmt_id"] = df["pmt_id"] + 1

    return df

def process(df):
    """
    In
    --
    df -> dataframe, ideally this should be the raw noise dataset.

    Out
    ---
    df -> (n, m+4) pandas dataframe with the positions and the label columns
    added, columns renamed and pmt_id changed to the local scheme.
    """
    df = add_positions(df)
    df = add_label(df)
    df = rename_columns(df)
    df = transform_pmt_id_scheme(df)

    return df
