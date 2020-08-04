from context import km3net
from km3net.utils import DATADIR
import km3net.data.pattern_matrix as pm
import sys

if __name__ == "__main__":
    path = DATADIR + '/processed/slice-615.csv'
    forward = 'n'
    frac = float(sys.argv[1])
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
    print("equalizing targets...")
    sample = pm.equalise_targets(sample)
    print("---")
    print("Shape of equalised sample: {0}".format(sample.shape))
    print("Shape of related: {0}".format(sample[sample['label'] == 1].shape))
    print("Shape of unrelated: {0}".format(sample[sample['label'] == 0].shape))

    save = input("save to disk? [y/n]: ")
    if save.lower() == 'y':
        sample.to_csv(DATADIR+'/train/slice-615-25-equal.csv', index=False,
                header=False)

    print("DONE")

