import os
from abc import ABC

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd

from repsys import Model
from repsys.helpers import set_seed
from repsys.ui import Select, Number


class BaseModel(Model, ABC):
    def _checkpoint_path(self):
        return os.path.join("./checkpoints", f"{self.name()}.npy")

    def _create_checkpoints_dir(self):
        dir_path = os.path.dirname(self._checkpoint_path())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def _mask_items(self, X_predict, item_indices):
        mask = np.ones(self.dataset.items.shape[0], dtype=np.bool)
        mask[item_indices] = 0
        X_predict[:, mask] = 0

    def _filter_items(self, X_predict, col, value):
        indices = self.dataset.filter_items_by_tags(col, [value])
        self._mask_items(X_predict, indices)

    def _apply_filters(self, X_predict, **kwargs):
        if kwargs.get("genre"):
            self._filter_items(X_predict, "genres", kwargs.get("genre"))

    def web_params(self):
        return {
            "genre": Select(options=self.dataset.tags.get("genres")),
        }


class KNN(BaseModel):
    def __init__(self, n: int = 50):
        self.model = NearestNeighbors(algorithm="brute", n_neighbors=n, metric="cosine")

    def name(self):
        return "knn"

    def fit(self, training=False):
        X = self.dataset.get_train_data()
        self.model.fit(X)

    def predict(self, X, **kwargs):
        if X.count_nonzero() == 0:
            return np.random.uniform(size=X.shape)

        if kwargs.get("neighbors"):
            distances, indices = self.model.kneighbors(X, n_neighbors=kwargs.get("neighbors"))
        else:
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

        vf = np.vectorize(f, signature="(n),(n)->(m)")
        X_predict = vf(n_distances, n_indices)

        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict

    def web_params(self):
        new_params = super(KNN, self).web_params()
        new_params["neighbors"] = Number()
        return new_params


class TopPopular(BaseModel):
    def __init__(self):
        self.item_ratings = None
        self.scaler = MinMaxScaler()

    def name(self) -> str:
        return "pop"

    def fit(self, training: bool = False) -> None:
        X = self.dataset.get_train_data()

        item_popularity = np.asarray((X > 0).sum(axis=0)).reshape(-1, 1)
        item_ratings = self.scaler.fit_transform(item_popularity)

        self.item_ratings = item_ratings.reshape(1, -1)

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = np.ones(X.shape)
        X_predict[X.nonzero()] = 0

        X_predict = X_predict * self.item_ratings

        self._apply_filters(X_predict, **kwargs)

        return X_predict


class Rand(BaseModel):
    def name(self) -> str:
        return "rand"

    def fit(self, training: bool = False) -> None:
        return

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = np.ones(X.shape)
        X_predict[X.nonzero()] = 0

        set_seed(self.config.seed)
        item_ratings = np.random.uniform(size=X.shape)
        X_predict = X_predict * item_ratings

        self._apply_filters(X_predict, **kwargs)

        return X_predict


class PureSVD(BaseModel):
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.sim = None

    def name(self) -> str:
        return "svd"

    def _save_model(self):
        self._create_checkpoints_dir()
        np.save(self._checkpoint_path(), self.sim)

    def _load_model(self):
        self.sim = np.load(self._checkpoint_path())

    def fit(self, training=False):
        X = self.dataset.get_train_data()

        U, sigma, VT = randomized_svd(X, self.n_factors, random_state=self.config.seed)
        self.sim = VT.T.dot(VT)

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.sim)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict


class EASE(BaseModel):
    def __init__(self, lmb: int = 100):
        self.sim = None
        self.lmb = lmb

    def name(self) -> str:
        return "ease"

    def _save_model(self):
        self._create_checkpoints_dir()
        np.save(self._checkpoint_path(), self.sim)

    def _load_model(self):
        self.sim = np.load(self._checkpoint_path())

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            G = X.T.dot(X).toarray()
            diagonal_indices = np.diag_indices(G.shape[0])
            G[diagonal_indices] += self.lmb
            P = np.linalg.inv(G)
            B = P / (-np.diag(P))
            B[diagonal_indices] = 0
            self.sim = B
            self._save_model()
        else:
            self._load_model()

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = X.dot(self.sim)
        X_predict[X.nonzero()] = 0

        self._apply_filters(X_predict, **kwargs)

        return X_predict
