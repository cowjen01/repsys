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

For ARM architecture, please install the Jax package using [Conda Forge](https://anaconda.org/conda-forge/jaxlib).

## Getting started

Before you begin, please create an empty folder that will contain the project's source code and add the following files:

```
├── __init__.py
├── dataset.py
├── models.py
├── repsys.ini
└── .gitignore
```

### dataset.py

Firstly we need to import our dataset. We will use [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) with 
20 million ratings made by 138,000 users to 27,000 movies for the tutorial purpose. Please download the `ml-20m.zip` file and unzip 
the data into the current folder. Then add the following content to the `dataset.py` file:

```python
import pandas as pd

from repsys import Dataset
import repsys.dtypes as dtypes

class MovieLens(Dataset):
    def name(self):
        return "ml20m"

    def item_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.Title(),
            "genres": dtypes.Tag(sep="|"),
            "year": dtypes.Number(data_type=int),
        }

    def interaction_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_json("./ml-20m/movies.csv")
        df["year"] = df["title"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        df = pd.read_csv("./ml-20m/ratings.csv")
        return df
```

This code will define a new dataset called ml20m, and it will import both ratings 
and items data. You must always specify your data structure using predefined data types.
Before you return the data, you can also preprocess it like extracting the movie's year from the title column.

### models.py
Now we define the first recommendation model, which will be a simple implementation of the user-based KNN.

```python
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from repsys import Model

class KNN(Model):
    def __init__(self):
        self.model = NearestNeighbors(n_neighbors=20, metric="cosine")

    def name(self):
        return "knn"
    
    def fit(self, training=False):
        X = self.dataset.get_train_data()
        self.model.fit(X)

    def predict(self, X, **kwargs):
        distances, indices = self.model.kneighbors(X)
        
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        distances = 1 - distances
        sums = distances.sum(axis=1)
        distances = distances / sums[:, np.newaxis]

        def f(dist, idx):
            A = self.dataset.get_train_data()[idx]
            D = sp.diags(dist)
            return D.dot(A).sum(axis=0)

        vf = np.vectorize(f, signature='(n),(n)->(m)')
       
        predictions = vf(distances, indices)
        predictions[X.nonzero()] = 0

        return predictions
```

You must define the fit method to train your model using the training data or load the previously trained model from a file.
All models are fitted when the web application starts, or the evaluation process begins.

If this is not a training phase, always load your model from a checkpoint to speed up the process. For tutorial purposes, this is omitted.

You must also define the prediction method that receives a sparse matrix of the users' interactions on the input. 
For each user (row of the matrix) and item (column of the matrix), the method should return a predicted score indicating 
how much the user will enjoy the item.

### repsys.ini

The last file we should create is a configuration that allows you to control a data splitting process, server settings, 
framework behavior, etc.

```ini
[general]
seed=123
debug=true

[dataset]
train_split_prop=0.8
test_holdout_prop=0.2
min_user_interacts=5
min_item_interacts=5

[server]
port=3001
```

### Splitting the data

Before we train our models, we need to split the data into train, validation, and test sets. Run the following command from the current directory.

```
$ repsys dataset split
```

This will hold out 80% of the users as training data, and the rest 20% will be used as validation/test data with 10% of users each. For both validation 
and test set, 20% of the interactions will also be held out for evaluation purposes. The split dataset will be stored in the default checkpoints folder.

### Training the models

Now we can move to the training process. To do this, please call the following command.

```
$ repsys model train
```

This command will call the fit method of each model with the training flag set to true.

## Sponsoring

The development of this framework is sponsored by the [Recombee](https://www.recombee.com) company.

![Recombee logo](./assets/recombee_logo.jpeg)
