import pandas as pd

p1_col_names = {'pos_x': 'x1', 'pos_y': 'y1',
                'pos_z': 'z1', 'time': 't1',
                'label': 'l1', 'event_id': 'eid1',
                'timeslice': 'ts1', 'id':'id1'}
p2_col_names = {'pos_x': 'x2', 'pos_y': 'y2',
                'pos_z': 'z2', 'time': 't2',
                'label': 'l2', 'event_id': 'eid2',
                'timeslice': 'ts2', 'id':'id2'}

col_names = ['x1', 'y1', 'z1', 't1', 'x2', 'y2', 'z2', 't2', 'label']

def add_label(df):
    """
    In
    --
    df -> dataframe, ideally should be the "exploded" frame.

    Out
    ---
    df -> dataframe, with label column added.
    """
    df['label'] = df['eid1'] == df['eid2']
    df['label'] = df['label'].astype(int)

    return df

def explode(df, model='pm'):
    """
    In
    --
    df -> (n, m) frame, ideally should be the processed dataset.

    Out
    ---
    df -> (sum(n-1, 1), m-1*2) frame, each row paired with all other unique
    rows.
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
        if model == 'gcd':
            dup = dup[dup['id2'] != dup['id1']]
        else:
            dup = dup[dup['id2'] > dup['id1']]

        result = pd.concat([result, dup])

    return result

def take_diff(df):
    """
    In
    --
    df -> (n, 9) dataframe, ideally should be the exploded dataframe.

    Out
    ---
    df -> (n, 5) dataframe, with only the difference of the x,y,z,t features.
    """
    df['x'] = df['x1'] - df['x2']
    df['y'] = df['y1'] - df['y2']
    df['z'] = df['z1'] - df['z2']
    df['t'] = df['t1'] - df['t2']

    return df

def process(df, drop=True, diff=True, model='pm'):
    """
    In
    --
    df -> dataframe, ideally should be a sample of the processed dataset.
    drop -> Bool, drop columns not required for training. They are: 'l1',
    'eid1', 'ts1', 'id1', 'l2', 'eid2', 'ts2', 'id2'.
    model -> Str, 'pm' or 'gcd'
    diff -> Bool, take difference of x,y,z,t features, defaults to True.

    Out
    ---
    df -> dataframe, 'exploded', related label added, unwanted columns
    dropped.
    """
    df = explode(df, model=model)
    df = add_label(df)
    if drop:
        df = df.drop(columns=['l1', 'eid1', 'ts1', 'id1', 'l2', 'eid2', 'ts2', 'id2'])
    if diff:
        df = take_diff(df)

    return df
