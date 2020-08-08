km3net.data module
------------------

This is the `km3net.data` module consisting of four sub modules namely
`km3net.data.noise`, `km3net.data.hits`, `km3net.data.data` and
`km3net.data.pattern_matrix`. The module is organized based on the different
types of datasets used and the various phases of the project. Each submodule
implements the `process()` method along with various helper methods that are
specific to the type of data handled. For example, the `km3net.data.noise`
submodule is intended to process the raw dataset containing hits from
background noise whilst `km3net.data.hits` is intended to process the HD5 file
containing hits from neutrino events. The `km3net.data.data.process` method is
provided as a oneshot creation of the `data/processed/data.csv` dataset however
one may also pick and choose the specific transformations to apply and generate
a dataset that fits their needs.

km3net.data.data module
=======================

The `km3net.data.data` module contains functions for preparation of
`data/processed/data.csv` which is the main dataset for this project consisting
of hits registered by the detector from both neutrino events and background
noise.

km3net.data.noise module
========================

The `km3net.data.noise` module contains functions for preparation of
`data/processed/noise.csv` which is the dataset containing hits from only
background noise.

km3net.data.hits module
=========================

The `km3net.data.hits` module contains functions for preparation of
`data/processed/events.csv` which is the dataset containing hits from only
neutrinos.

km3net.data.pattern_matrix module
=================================

The `km3net.data.pattern_matrix` module contains functions for preparation of
datasets to train the corresponding model to replace the *Pattern Matrix* step
of the existing pipeline.

