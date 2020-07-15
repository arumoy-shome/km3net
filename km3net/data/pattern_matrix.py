import pandas as pd
from km3net.utils import DATADIR

DATAFILE = DATADIR + "/train/slice-615.csv"
p1_col_names = {'pos_x': 'x1', 'pos_y': 'y1',
                'pos_z': 'z1', 'time': 't1',
                'label': 'l1', 'event_id': 'eid1',
                'timeslice': 'ts1', 'id':'id1'}
p2_col_names = {'pos_x': 'x2', 'pos_y': 'y2',
                'pos_z': 'z2', 'time': 't2',
                'label': 'l2', 'event_id': 'eid2',
                'timeslice': 'ts2', 'id':'id2'}

def load(file=DATAFILE):
    """
    In: file -> Str, csv file to load.
    Out: frame representing the training data.
    Expectations: expects `file` to be a csv file without headers.
    """
    if os.path.file(file):
        return pd.read_csv(file)
    else:
        print('{0} does not exist.'.format(file))

def add_label(df):
    """
    In: df -> frame, should be the exploded frame.
    Out: frame, with label column added.
    """
    df['label'] = df['eid1'] == df['eid2']
    df['label'] = df['label'].astype(int)

    return df

def explode(df):
    """
    In: df -> (n, m) frame, ideally should be the processed dataset.
    Out: (sum(n, 0), m*2) frame, each row paired with all other unique rows.
    """
    result = pd.DataFrame()
    df['id'] = df.reset_index().index

    for id, row in df.iterrows():
        dup = df.copy()

         # may be a better way to do this
        dup['pos_x'] = row['pos_x']
        dup['pos_y'] = row['pos_y']
        dup['pos_z'] = row['pos_z']
        dup['time'] = row['time']
        dup['label'] = row['label']
        dup['event_id'] = row['event_id']
        dup['timeslice'] = row['timeslice']
        dup['id'] = row['id']

        dup = dup.rename(columns=p1_col_names)
        dup = pd.concat([dup, df], axis=1)
        dup = dup.rename(columns=p2_col_names)
        dup = dup[dup['id2'] > dup['id1']]
        result = pd.concat([result, dup])

    return result

def process(df, drop=True, write=False):
    """
    In: df -> frame, ideally should be a sample of the processed dataset.
    drop -> Bool, drop columns not required for training. They are: 'l1',
    'eid1', 'ts1', 'id1', 'l2', 'eid2', 'ts2', 'id2'.
    write -> Truthy, if string of len > 0 then use it as filename and save
    frame to disk. Note that the filename should be relative to DATADIR. Use
    km3net.utils.DATADIR to get the correct absolute path.
    Out: frame, 'exploded', related label added, unwanted columns dropped.
    """
    df = explode(df)
    df = add_label(df)
    df = df.drop(columns=['l1', 'eid1', 'ts1', 'id1', 'l2', 'eid2', 'ts2', 'id2'])

    if write and len(write) > 0:
        df.to_csv(DATADIR + write, index=False, header=False)

    return df
