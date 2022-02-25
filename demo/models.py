import os
import pickle
from abc import ABC

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import repsys.web as web
from repsys import Model


# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/


class BaseModel(Model, ABC):
    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.name())

    def _serialize(self):
        return self.model

    def _deserialize(self, checkpoint):
        self.model = checkpoint

    def _load_model(self):
        checkpoint = pickle.load(open(self._checkpoint_path(), "rb"))
        self._deserialize(checkpoint)

    def _filter_genres(self, predictions, genre):
        items = self.dataset.items
        exclude_ids = items.index[items["genre"].apply(lambda genres: genre not in genres)]
        exclude_indexes = exclude_ids.map(self.dataset.item_id_to_index)
        predictions[:, exclude_indexes] = 0

    def _save_model(self):
        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self._serialize(), checkpoint)

    def web_params(self):
        return {
            'genre': web.Select(options=self.dataset.get_genres()),
        }


class EASE(BaseModel):
    def __init__(self, l2_lambda=0.5):
        super().__init__()
        self.B = None
        self.l2_lambda = l2_lambda

    def name(self) -> str:
        return "ease"

    def _serialize(self):
        return self.B

    def _deserialize(self, checkpoint):
        self.B = checkpoint

    def fit(self, training: bool = False) -> None:
        if training:
            X = self.dataset.get_train_data()
            G = X.T.dot(X).toarray()

            diagonal_indices = np.diag_indices(G.shape[0])
            G[diagonal_indices] += self.l2_lambda

            P = np.linalg.inv(G)
            B = P / (-np.diag(P))
            B[diagonal_indices] = 0

            self.B = B
            self._save_model()
        else:
            self._load_model()

    def predict(self, x: csr_matrix, **kwargs):
        predictions = x.dot(self.B)

        predictions[x.nonzero()] = 0

        if kwargs.get("genre"):
            self._filter_genres(predictions, kwargs.get("genre"))

        return predictions


class KNN(BaseModel):
    def __init__(self, k=15):
        super().__init__()
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def name(self):
        return "knn"

    def fit(self, training=False):
        if training:
            self.model.fit(self.dataset.get_train_data())
            self._save_model()
        else:
            self._load_model()

    def predict(self, x, **kwargs):
        # the slowest phase of the prediction
        distances, indexes = self.model.kneighbors(x)

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
        predictions[x.nonzero()] = 0

        if kwargs.get("genre"):
            self._filter_genres(predictions, kwargs.get("genre"))

        return predictions
