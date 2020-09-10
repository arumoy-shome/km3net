from context import km3net
from km3net.utils import DATADIR
from km3net.data.model import process
from km3net.data.utils import equalise_targets
import pandas as pd
import sys
import os

if __name__ == "__main__":
    forward = 'n'
    path = DATADIR + input('source data file: ')
    prefix = DATADIR + input('prefix for the final data file(s): ')
    equalise = input('equalise targets? [y/n]: ')
    frac = float(input('frac: '))
    save = input('save to disk? [y/n]: ')

    while not os.path.isfile(path):
        print('{0} does not exist!'.format(path))
        path = DATADIR+input('source data file: ')

    print("source file: {0}".format(path))
    print("final file prefix: {0}".format(prefix))
    print("frac: {0}".format(frac))
    master = input("proceed? [y/n]: ")

    if master.lower() == 'n': exit()

    df = pd.read_csv(path)

    # make sure the distribution is acceptable before proceeding
    while forward.lower() != 'y':
        sample = df.sample(frac=frac)
        print("---")
        print("Shape of sample: {0}".format(sample.shape))
        print("Shape of hits: {0}".format(sample[sample['label'] == 1].shape))
        print("Shape of noise: {0}".format(sample[sample['label'] == 0].shape))
        forward = input("proceed? [y/n]: ")

    # equalize tragets
    if equalise.lower() == 'y':
        print("equalizing targets...")
        sample = equalise_targets(sample)
        print("---")
        print("Shape of equalised sample: {0}".format(sample.shape))
        print("Shape of related: {0}".format(sample[sample['label'] == 1].shape))
        print("Shape of unrelated: {0}".format(sample[sample['label'] == 0].shape))

    print("saving {}".format(prefix+'-equal.csv'))
    sample.to_csv(prefix+'-equal.csv', index=False, header=False)

    # create pair-wise dataset
    print("creating pair-wise dataset...")
    sample = process(sample)
    print("---")
    print("Shape of processed sample: {0}".format(sample.shape))
    print("Shape of related: {0}".format(sample[sample['label'] == 1].shape))
    print("Shape of unrelated: {0}".format(sample[sample['label'] == 0].shape))

    print("saving {}".format(prefix+'-explode.csv'))
    sample.to_csv(prefix+'-explode.csv', index=False, header=False)

    print("DONE")
