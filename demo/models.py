import os
import pickle
from abc import ABC

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from repsys import Model
from repsys.ui import Select


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

    def _save_model(self):
        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self._serialize(), checkpoint)

    def _apply_filters(self, predictions, **kwargs):
        if kwargs.get("genre"):
            selected_genre = kwargs.get("genre")
            items = self.dataset.items
            exclude_ids = items.index[items["genres"].apply(lambda genres: selected_genre not in genres)]
            exclude_indices = exclude_ids.map(self.dataset.item_id_to_index)
            predictions[:, exclude_indices] = 0

    def web_params(self):
        return {
            'genre': Select(options=self.dataset.tags.get('genres')),
        }


class KNN(BaseModel):
    def __init__(self):
        self.model = NearestNeighbors(algorithm='brute', n_neighbors=20, metric='cosine')

    def name(self):
        return "knn"

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            self.model.fit(X)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X, **kwargs):
        distances, indices = self.model.kneighbors(X)

        n_distances = distances[:, 1:]
        n_indices = indices[:, 1:]

        n_distances = 1 - n_distances

        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        def f(dist, idx):
            A = self.dataset.get_train_data()[idx]
            D = sp.diags(dist)
            return D.dot(A).sum(axis=0)

        vf = np.vectorize(f, signature='(n),(n)->(m)')
        predictions = vf(n_distances, n_indices)

        predictions[X.nonzero()] = 0

        self._apply_filters(predictions, **kwargs)

        return predictions


class EASE(BaseModel):
    def __init__(self, l2_lambda=0.5):
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

    def predict(self, X: csr_matrix, **kwargs):
        predictions = X.dot(self.B)
        predictions[X.nonzero()] = 0

        self._apply_filters(predictions, **kwargs)

        return predictions
