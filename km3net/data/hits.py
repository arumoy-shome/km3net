import pandas as pd
import os

DATAFILE = "data/raw/events.h5"

def load():
    """
    In: None
    Out: frame, frame tuple representing the mc_info and mc_hits
    tables respectively. None if expectations are not met.
    Expects: 'data/raw/events.h5' to exist.
    """
    if os.path.isfile(DATAFILE):
        return pd.read_hdf(DATAFILE, key='/data/mc_hits')
    else:
        print('{0} does not exist.'.format(DATAFILE))

def add_label(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw hits dataset.
    Out: (n, m+1) pandas dataframe with one new column representing the label.
    """
    df["label"] = 1

    return df

def add_event_id(df):
    """
    In: df -> (n, m) pandas dataframe, this should be the raw hits dataset (mc_hits).
    Expects: 'data/raw/events.h5' to be a valid file path.
    Out: (n, m+1) hits with their corresponding event id added as a new column.
    """
    info = pd.read_hdf("data/raw/events.h5", key="/data/mc_info")
    info["event_id"] = info.index
    hits["id"] = hits.index
    info = info.drop_duplicates(subset='nu.hits.end')
    bins = pd.concat([pd.Series([0]), info["nu.hits.end"]])
    hits["event_id"] = pd.cut(hits.id, bins=bins, right=False,
    labels=info["event_id"], include_lowest=True)
    hits = hits.drop(columns=['id'])

    return hits

def rename_columns(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw hits dataset.
    Out: (n, m) pandas dataframe with the columns renamed in snakecase.
    """
    df = df.rename(columns={'h.dom_id': 'dom_id', 'h.pmt_id':
    'pmt_id', 'h.pos.x': 'pos_x', 'h.pos.y': 'pos_y', 'h.pos.z':
    'pos_z', 'h.dir.x': 'dir_x', 'h.dir.y': 'dir_y', 'h.dir.z':
    'dir_z', 'h.tot': 'tot', 'h.t': 'time'})

    return df

def transform_pmt_id_scheme(df):
    """
    In: df -> (n, m) pandas dataframe, this should ideally be the raw
    hits dataset where the pmt_id column follows the global numbering
    scheme.
    Out: (n, m) pandas dataframe with pmt_id column now following the
    local numbering scheme.
    """
    df["pmt_id"] = df["pmt_id"] - 31 * (df["dom_id"] - 1)

    return df

def process(write=False):
    """
    In: write -> Bool, write frame to disk as
    'data/processed/hits.csv'. Defaults to False.
    Out: (n, m+2) pandas dataframe with the label and event_id columns
    added, columns renamed and pmt_id changed to the local scheme.
    """
    df = load()
    df = rename_columns(df)
    df = add_label(df)
    df = add_event_id(df)
    df = transform_pmt_id_scheme(df)

    return df
