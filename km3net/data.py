"""
This module contains functions to procress the raw noise dataset into
a form more suitable for analysis. Although the model exposes several
methods, it is intended to be used by envoking the process_noise
method.
"""

import pandas as pd
import numpy as np
import re

def add_noise_pos(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise
    dataset without any position info.
    Expects data/noise.detx to be a valid file path containing position coordinates of noise hits.
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
    Out: (n, m+1) pandas dataset with one new column representing the label.
    """
    df["label"] = 0

    return df

def rename_noise_col(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m) pandas dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'dom id': 'dom_id', 'pmt id': 'pmt_id',
    'time-over-threshold': 'tot', 'x': 'pos_x', 'y': 'pos_y', 'z':
    'pos_z', 'dx': 'dir_x', 'dy': 'dir_y', 'dz': 'dir_z'})

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

def process_noise(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw noise dataset.
    Out: (n, m+4) pandas dataframe with the positions and the label added.
    """
    df = add_noise_pos(df)
    df = add_noise_label(df)
    df = rename_noise_col(df)

