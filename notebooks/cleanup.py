import pandas as pd

"""
# load the dataset
"""
events = pd.read_hdf("data/events.h5")
noise = pd.read_csv("data/noise.csv")

"""
# expore `noise`
"""
noise.shape
noise.head

"""
make sure there are no missing values (there aren't any)
"""
noise.isna().any().any()

"""
# check if noise.time is unique (**it isn't**)
NOTE this method takes a good few seconds to run
NOTE so we cannot use it as index column
NOTE it makes sense why it's not unique, we may have hits in differnt DOMs at
the same time
"""
# noise.time.is_unique

"""
# explore `events`
`events` contains 18 columns however the each item is inside a (rather
redundant) list, so we first flatten it.
"""

events = events.applymap(lambda x: x[0])
events.head()
events.shape

"""
save this for future work
"""
events.to_csv("data/events.csv", index=False)
