from context import km3net
from km3net.utils import DATADIR
import km3net.data.pattern_matrix as pm
import pandas as pd
import sys
import os

if __name__ == "__main__":
    forward = 'n'
    path = DATADIR + sys.argv[1]
    frac = float(sys.argv[2])

    while not os.path.isfile(path):
        print('{0} does not exist!'.format(path))
        path = DATADIR+input('path? :')

    print("data file: {0}, frac: {1}".format(path, frac))
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

    # create pair-wise dataset
    print("creating pair-wise dataset...")
    sample = pm.process(sample)
    print("---")
    print("Shape of processed sample: {0}".format(sample.shape))
    print("Shape of related: {0}".format(sample[sample['label'] == 1].shape))
    print("Shape of unrelated: {0}".format(sample[sample['label'] == 0].shape))

    # equalize tragets
    equalise = input("equalise targets? [y/n]: ")
    if equalise.lower() == 'y':
        print("equalizing targets...")
        sample = pm.equalise_targets(sample)
        print("---")
        print("Shape of equalised sample: {0}".format(sample.shape))
        print("Shape of related: {0}".format(sample[sample['label'] == 1].shape))
        print("Shape of unrelated: {0}".format(sample[sample['label'] == 0].shape))

    save = input("save to disk? [y/n]: ")
    if save.lower() == 'y':
        name = input("name? :")
        sample.to_csv(DATADIR+name, index=False, header=False)

    print("DONE")

