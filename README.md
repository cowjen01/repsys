# Repsys

[![PyPI version](https://badge.fury.io/py/repsys-framework.svg)](https://badge.fury.io/py/repsys-framework)

The Repsys is a framework for developing and analyzing recommendation systems, and it allows you to:
- Add your own dataset and recommendation models
- Visually evaluate the models on various metrics
- Quickly create dataset embeddings to explore the data
- Preview recommendations using a web application
- Simulate user's behavior while receiving the recommendations

![web preview](https://github.com/cowjen01/repsys/raw/master/images/web-preview.jpg)

<p align="middle">
  <img src="https://github.com/cowjen01/repsys/raw/master/images/dataset-eval.jpg" width="48%" />
  <img src="https://github.com/cowjen01/repsys/raw/master/images/model-eval.jpg" width="48%" /> 
</p>

## Installation

Install the package using [pip](https://pypi.org/project/repsys-framework/):

```
$ pip install repsys-framework
```

## Getting started

If you want to skip this tutorial and try the framework, you can pull the content of the [demo](https://github.com/cowjen01/repsys/tree/master/demo) folder located at the repository.
As mentioned in the [next step](https://github.com/cowjen01/repsys#datasetpy), you still have to download the dataset before you begin.

Otherwise, please create an empty project folder that will contain the dataset and models implementation.

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
        df = pd.read_csv("./ml-20m/movies.csv")
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
from repsys.ui import Select

class KNN(Model):
    def __init__(self):
        self.model = NearestNeighbors(n_neighbors=20, metric="cosine")

    def name(self):
        return "knn"
    
    def fit(self, training=False):
        X = self.dataset.get_train_data()
        self.model.fit(X)

    def predict(self, X, **kwargs):
        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)
    
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

        vf = np.vectorize(f, signature="(n),(n)->(m)")
       
        predictions = vf(distances, indices)
        predictions[X.nonzero()] = 0
        
        if kwargs.get("genre"):
            items = self.dataset.items
            items = items[items["genres"].apply(lambda x: kwargs.get("genre") not in x)]
            indices = items.index.map(self.dataset.item_id_to_index)
            predictions[:, indices] = 0

        return predictions

    def web_params(self):
        return {
            "genre": Select(options=self.dataset.tags.get("genres")),
        }
```

You must define the fit method to train your model using the training data or load the previously trained model from a file.
All models are fitted when the web application starts, or the evaluation process begins. If this is not a training phase, always 
load your model from a checkpoint to speed up the process. For tutorial purposes, this is omitted.

You must also define the prediction method that receives a sparse matrix of the users' interactions on the input. 
For each user (row of the matrix) and item (column of the matrix), the method should return a predicted score indicating 
how much the user will enjoy the item.

Additionally, you can specify some web application parameters you can set during recommender creation. The value is then accessible in 
the `**kwargs` argument of the prediction method.  In the example, we create a select input with all unique genres and filter out only those movies that do not contain the selected genre. 

### repsys.ini

The last file we should create is a configuration that allows you to control a data splitting process, server settings, 
framework behavior, etc.

```ini
[general]
seed=123

[dataset]
train_split_prop=0.85
test_holdout_prop=0.2
min_user_interacts=5
min_item_interacts=5

[evaluation]
recall_precision_k=20,50
ndcg_k=100
coverage_k=20,50

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

This command will call the fit method of each model with the training flag set to true. You can always limit the models using `-m` flag with the model's name as a parameter.


### Evaluating the models

When the data is prepared and the models trained, we can evaluate the performance of the models on the unseen users' interactions. Run the following command to do so.

```
$ repsys model eval
```

Again, you can limit the models using the `-m` flag. The results will be stored in the checkpoints folder when the evaluation is done.

### Evaluating the dataset

Before starting the web application, the final step is to evaluate the dataset's data. This procedure will create users and items embeddings of the training and validation data 
to allow you to explore the latent space. Run the following command from the project directory.

```
$ repsys dataset eval
```

You can choose from three types of embeddings algorithm:
1. [PyMDE](https://pymde.org) (Minimum-Distortion Embedding) is a fast library designed to distort relationships between pairs of items minimally. Use `--method pymde` (this is the default option).
2. Combination of the PCA and TSNE algorithms (reduction of the dimensionality to 50 using PCA, then reduction to 2D space using TSNE). Use `--method tsne`.
3. Your own implementation of the algorithm. Use `--method custom` and add the following method to the model's class of your choice. In this case, you must also specify the model's name using `-m` parameter.

```python
from sklearn.decomposition import NMF

def compute_embeddings(self, X):
    nmf = NMF(n_components=2)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H.T
```

In the example, the negative matrix factorization is used. You have to return a user and item embeddings pair in this order. Also, it is essential to return the matrices in the shape of (n_users/n_items, n_dim). 
If the reduced dimension is higher than 2, the TSNE method is applied.

### Running the application

Finally, it is time to start the web application to see the results of the evaluations and preview live recommendations of your models.

```
$ repsys server
```

The application should be accessible on the default address [http://localhost:3001](http://localhost:3001). When you open the link, you will see the main screen where your recommendations appear once you finish the setup.
The first step is defining how the items' data columns should be mapped to the item view components.

![app setup](https://github.com/cowjen01/repsys/raw/master/images/app-setup.jpg)

Then we need to switch to the build mode and add two recommenders - one without filter and the second with only comedy movies included.

![add recommender](https://github.com/cowjen01/repsys/raw/master/images/add-recommender.jpg)

Now we switch back from the build mode and select a user from the validation set (never seen by a model before).

![user select](https://github.com/cowjen01/repsys/raw/master/images/user-selection.jpg)

Finally, we see the user's interaction history on the right side and the recommendations made by the model on the left side.

![user select](https://github.com/cowjen01/repsys/raw/master/images/recoms-preview.jpg)

## Contribution

To build the package from the source, you first need to install Node.js and npm library as documented [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).
Then you can run the following script from the root directory to build the web application and install the package locally.

```
$ ./scripts/install-locally.sh
```

## Sponsoring

The development of this framework is sponsored by the [Recombee](https://www.recombee.com) company.

<img src="https://github.com/cowjen01/repsys/raw/master/images/recombee-logo.png" width="50%" />
