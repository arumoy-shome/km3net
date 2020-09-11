"""This script generates data/{train,test}/gcd/*.csv files used to
train the NodeNet model to perform node classification. The script
prompts the user to input the preferred number of nodes N for the
graph and the number of nodes per class are equalized (assumes binary
classification). The output is two csv files in the specified
directory. The $prefix.csv contains (N,5) feature matrix where the
first 4 columns are the node features and the last column is the node
label. The $prefix-weights.csv file contains is a (N**2-N,) vector
containing the edge weights of the graph (assumes a fully connected
graph).

"""

from context import km3net
from km3net.utils import DATADIR, yes
from km3net.data.model import process
import pandas as pd
import sys
import os

if __name__ == "__main__":
    forward = 'n'
    path = DATADIR + input('source data file: ')
    prefix = DATADIR + input('prefix for the final data file(s): ')
    num_nodes = int(input('number of nodes?: '))
    equalise = input("equalise targets? [y/n]: ")

    while not os.path.isfile(path):
        print('{0} does not exist!'.format(path))
        path = DATADIR+input('source data file: ')

    print("source file: {0}".format(path))
    print("final file names: {0} and {1}".format(prefix+'.csv', prefix+'-weights.csv'))
    print("number of nodes: {0}".format(num_nodes))
    print("equalise targets: {}".format(equalise))
    master = input("proceed? [y/n]: ")

    if not yes(master): exit()

    df = pd.read_csv(path)
    noise = df[df.label == 0]
    events = df[df.label == 1]

    if yes(equalise):
        esample = events.sample(num_nodes//2)
        nsample = noise.sample(num_nodes//2)
        sample = pd.concat([esample, nsample])
    else:
        # make sure the distribution is acceptable before proceeding
        while not yes(forward):
            sample = df.sample(num_nodes)
            print("---")
            print("Shape of sample: {0}".format(sample.shape))
            print("Shape of events: {0}".format(sample[sample['label'] == 1].shape))
            print("Shape of noise: {0}".format(sample[sample['label'] == 0].shape))
            forward = input("proceed? [y/n]: ")

    X = sample[['pos_x', 'pos_y', 'pos_z', 'time', 'label']]
    weights = process(sample, model='gcd')['label']

    print("saving {}".format(prefix+'.csv'))
    X.to_csv(prefix+'.csv', index=False, header=False)

    print("saving {}".format(prefix+'-weights.csv'))
    weights.to_csv(prefix+'-weights.csv', index=False, header=False)

    print("DONE")
