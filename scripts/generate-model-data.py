"""This script generates data/{train,test}/{gcd,pm}/*.csv files used
to train and test the models. The script prompts the user for the
following parameters:

1. source data file: the csv file from which the output file(s) will
be derived. This can be the main dataset or a timeslice, with the
x,y,z,t feature columns.
2. output file prefix: string to prepend to the output files, should
be relative to the DATADIR(see km3net.utils.DATADIR).
3. equalise: Bool to determine if the number of examples per class
should be equalised or not, if 'y' then the majority class is
undersampled to match the number of minority class samples
4. frac: Float to determine if a random sample from the source data
should be taken, set to 1.0 if sample should not be taken.
5. model: String to identify the model for which the data is being
generated for, set to 'pm' or 'gcd', see km3net.data.model.explode()
for more details.
6. diff: Bool to determine if the difference between the x,y,z,t
features should be taken

The script generates and saves the following csv files:

1. ${prefix}.csv: the data after taking the sample.
2. ${prefix}-processed.csv: the output of km3net.data.model.process().
"""

from context import km3net
from km3net.utils import DATADIR, yes
import km3net.data.model as data
import km3net.data.utils as utils
import pandas as pd
import sys
import os

def take_sample(df, frac):
    df = df.sample(frac=frac)
    print_stats('sampled', df)

    return df


def equalise_targets(df):
    print("equalising targets...")
    df = utils.equalise_targets(df)
    print_stats('equalised', df)

    return df


def process(df, model, diff):
    print("creating pair-wise dataset...")
    df = data.process(df, model=model, diff=diff)
    print_stats('processed', df)

    return df


def print_stats(prefix, df):
    print("---")
    print("Shape of {} data: {}".format(prefix, df.shape))
    print("Shape of positive class: {0}".format(df[df['label'] == 1].shape))
    print("Shape of negative class: {0}".format(df[df['label'] == 0].shape))


def save(sample, processed):
    print("saving {}".format(prefix+'.csv'))
    sample.to_csv(prefix+'.csv', index=False, header=False)

    print("saving {}".format(prefix+'-processed.csv'))
    processed.to_csv(prefix+'-processed.csv', index=False, header=False)


if __name__ == "__main__":
    forward = 'n'
    path = DATADIR + input('source data file: ')
    prefix = DATADIR + input('prefix for final data file(s): ')
    equalise = input("equalise targets? [y/n]: ")
    frac = float(input('frac: '))
    model = input('model ["pm"/"gcd"]: ')
    diff = input('take diff? [y/n]: ')
    diff = True if yes(diff) else False

    while not os.path.isfile(path):
        print('{0} does not exist!'.format(path))
        path = DATADIR+input('source data file: ')

    # print overview
    print("source file: {0}".format(path))
    print("final file(s): {0} and {1}".format(prefix+'.csv',
                                              prefix+'-processed.csv'))
    print("frac: {0}".format(frac))
    print("model: {}".format(model))
    print("diff: {}".format(diff))
    master = input("proceed? [y/n]: ")

    if not yes(master): exit()

    df = pd.read_csv(path)

    # make sure the distribution is acceptable before proceeding
    while not yes(forward):
        sample = take_sample(df, frac=frac)
        forward = input("proceed? [y/n]: ")

    # if data for gcd, equalise early
    if model == 'gcd':
        if yes(equalise):
            sample = equalise_targets(sample)

        # create pair-wise dataset
        processed = process(sample, model=model, diff=diff)
    else:
        # create pair-wise dataset
        processed = process(sample, model=model, diff=diff)

        # equalize targets later if data for pm
        if yes(equalise):
            processed = equalise_targets(processed)

    sample = sample[['pos_x', 'pos_y', 'pos_z', 'time', 'label']]

    save(sample, processed)

    print("DONE")

