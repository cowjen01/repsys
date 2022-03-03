# Repsys

The Repsys is a framework for developing and analyzing recommendation systems, and it allows you to:
- Add your own dataset and recommendation models
- Visually evaluate the models on various metrics
- Quickly create dataset embeddings to explore the data
- Preview recommendations using a web application
- Simulate user's behavior while receiving the recommendations

## Installation

Install the packages using [pip](https://pypi.org/project/pip/):

```
$ pip install repsys
```

The framework uses the [Jax](https://jax.readthedocs.io/en/latest/) library to speed up the computation of models 
evaluation by allowing to run these processes on GPU. The CPU version of the library is a part of the framework package.
To use the CUDA version, please follow [this guide](https://github.com/google/jax#pip-installation-gpu-cuda). 

```
$ pip install --upgrade pip
$ pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For M1 chips support, please install the CPU/CUDA package using [Conda Forge](https://anaconda.org/conda-forge/jaxlib), which currently contains the ARM version.

## Getting started

Before you begin, please create an empty folder that will contain the project's source code and create the following files:

```
├── __init__.py
├── dataset.py
├── models.py
├── repsys.ini
└── .gitignore
```

## Support

The development and testing of this framework are supported by the [Recombee](https://www.recombee.com) company.

![Recombee logo](./assets/recombee_logo.jpeg "Recombee")
