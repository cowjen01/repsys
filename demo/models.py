import os
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import repsys.web as web
from repsys import Model


# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/


class KNN(Model):
    def __init__(self, k=15):
        super().__init__()
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def name(self):
        return "knn"

    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.name())

    def _serialize(self):
        return self.model

    def _deserialize(self, checkpoint):
        self.model = checkpoint

    def _load_model(self):
        checkpoint = pickle.load(open(self._checkpoint_path(), "rb"))
        self._deserialize(checkpoint)

    def _save_model(self):
        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self._serialize(), checkpoint)

    def fit(self, training=False):
        if training:
            self.model.fit(self.dataset.get_train_data())
            self._save_model()
        else:
            self._load_model()

    def predict(self, X, **kwargs):
        # the slowest phase of the prediction
        distances, indexes = self.model.kneighbors(X)

        n_distances = distances[:, 1:]
        n_indexes = indexes[:, 1:]

        n_distances = 1 - n_distances

        # normalize distances
        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        def f(dist, idx):
            A = self.dataset.get_train_data()[idx]
            D = sp.diags(dist)
            return D.dot(A).sum(axis=0)

        # over users multiply the interactions by a distance
        vf = np.vectorize(f, signature='(n),(n)->(m)')
        predictions = vf(n_distances, n_indexes)

        # keep just the never seen items
        predictions[(X > 0).toarray()] = 0

        if kwargs.get("genre"):
            genre = kwargs.get("genre")
            items = self.dataset.items

            # get ids of the items don't contain the genre
            exclude_ids = items.index[items["genre"].apply(lambda x: genre not in x)]
            exclude_indexes = exclude_ids.map(self.dataset.item_id_to_index)

            # keep just the items with the genre
            predictions[:, exclude_indexes] = 0

        return predictions

    def web_params(self):
        return {
            'genre': web.Select(options=self.dataset.get_genres()),
            'exclude': web.Checkbox(default=True),
            'neighbors': web.Number(default=10)
        }
