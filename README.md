# KM3NeT

The [KM3NeT neutrino telescope](https://www.km3net.org/) is a research infrastructure being built on the sea bed of the Mediterranian to study neutrinos originating from cosmic events. In conjuncture with [The Netherlands eScience Centre](https://www.esciencecenter.nl/) and [Nikhef](https://www.nikhef.nl/), I conducted research to improve the existing data processing pipeline using Deep Learning Models. The project started in June 2020 and concluded in October 2020 with the goal to explore the following research questions:

1. Can the existing data processing pipeline implemented by Karas et al. [karas] be replaced by Deep Learning models?
2. Specifically, can the *Pattern Matrix* step of the pipeline be replaced by a Multi Layer Perceptron?
3. And, can the *Graph Community Detection* step be replaced by a Graph Convolutional Neural network?

Since the research conducted in this project is a direct improvement upon the Data Processing Pipeline implemented by Karas et al. [karas], I recommend reading their paper for more background. Details on the research and outcomes of this project can be found in the PDF document `report/thesis.pdf`.

[karas]: Karaś, K. (2019). Data processing pipeline for the KM3NeT neutrino telescope (Doctoral dissertation, Universiteit van Amsterdam).

## Project structure
The directory structure of the project along with a brief description is presented below:

```
|-root/         # the directory where this README.md is located
|--km3net/      # root python module
|----data/      # data module, contains code for preparation of data for the models
|----model/     # model module, contains code for creation, training and evaluation of models
|--data/        # root directory for storing data
|----raw/       # the raw datasets, immutable and source of truth
|----processed/ # itermediate datasets
|----train/     # datasets for training models
|----test/      # datasets for testing models
|--notebooks/   # Jupyter notebooks for experiments
|--report/      # Latex report
|--scripts/     # scripts for one off tasks, such as creation of datasets
```

The intended way of using this codebase is through Jupyter notebooks since it makes it easy to view code, visualizations and text in a single view. Further documentation for each module can be found in the python files and in the method doc strings. Please view one of the notebooks for example usage.
