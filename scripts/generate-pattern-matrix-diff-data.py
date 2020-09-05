from context import km3net
from km3net.utils import DATADIR
import km3net.data.pm as pm
import pandas as pd
import sys
import os

if __name__ == "__main__":
    path = DATADIR + input('source data file: ')
    header = input('does the source data file contain headers? [y/n]: ')
    name = DATADIR + input('name of final data file: ')
    save = input("save to disk? [y/n]: ")

    while not os.path.isfile(path):
        print('{0} does not exist!'.format(path))
        path = DATADIR+input('path? :')

    print("source file: {0}".format(path))
    print("final file: {0}".format(name))
    master = input("proceed? [y/n]: ")

    if master.lower() == 'n': exit()

    if header.lower() == 'y':
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=pm.col_names)


    print("---")
    print("Shape of df: {0}".format(df.shape))
    df['x'] = df['x1'] - df['x2']
    df['y'] = df['y1'] - df['y2']
    df['z'] = df['z1'] - df['z2']
    df['t'] = df['t1'] - df['t2']
    df = df.drop(columns=['x1', 'x2', 'y1', 'y2', 'z1', 'z2', 't1', 't2'])
    df = df[['x', 'y', 'z', 't', 'label']]
    print("Shape of diff df: {0}".format(df.shape))

    if save.lower() == 'y':
        df.to_csv(name, index=False, header=False)

    print("DONE")
