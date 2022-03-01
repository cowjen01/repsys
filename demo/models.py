import os
import pickle
from abc import ABC

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

import repsys.web as web
from repsys import Model


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
            genre = kwargs.get("genre")
            items = self.dataset.items

            exclude_ids = items.index[items["genre"].apply(lambda genres: genre not in genres)]
            exclude_indexes = exclude_ids.map(self.dataset.item_id_to_index)

            predictions[:, exclude_indexes] = 0

        if kwargs.get('category'):
            category = kwargs.get('category')
            items = self.dataset.items

            exclude_ids = items.index[items['product_type'] != category]
            exclude_indexes = exclude_ids.map(self.dataset.item_id_to_index)

            predictions[:, exclude_indexes] = 0

    def web_params(self):
        return {
            'category': web.Select(options=self.dataset.categories.get('product_type')),
            # 'genre': web.Select(options=self.dataset.tags('genre')),
        }


class KNN(BaseModel):
    def __init__(self, k=30):
        super().__init__()
        self.model = NearestNeighbors(n_neighbors=k, radius=1.5, metric="cosine")

    def name(self):
        return "knn"

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            self.model.fit(X)
            self._save_model()
        else:
            self._load_model()

    def _predict_knn(self, x: csr_matrix):
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

        return predictions

    def predict(self, x, **kwargs):
        predictions = self._predict_knn(x)
        predictions[x.nonzero()] = 0

        self._apply_filters(predictions, **kwargs)

        return predictions


class MF(KNN):
    def __init__(self, n=15):
        super().__init__()
        self.nmf = NMF(n_components=n, init='nndsvd', max_iter=100, random_state=0, verbose=2)

    def name(self):
        return "mf"

    def _serialize(self):
        return {
            'nmf': self.nmf,
            'knn': self.model
        }

    def _deserialize(self, checkpoint):
        self.nmf = checkpoint.get('nmf')
        self.model = checkpoint.get('knn')

    def fit(self, training=False):
        if training:
            X = self.dataset.get_train_data()
            W = self.nmf.fit_transform(X)
            self.model.fit(W)
            self._save_model()
        else:
            self._load_model()

    def predict(self, x: csr_matrix, **kwargs):
        W = self.nmf.transform(x)
        predictions = self._predict_knn(W)
        predictions[x.nonzero()] = 0

        self._apply_filters(predictions, **kwargs)

        return predictions


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

        self._apply_filters(predictions, **kwargs)

        return predictions


# class VAE(BaseModel):
#     def __init__(self):
#         super().__init__()
#         self.model = None
#
#     def name(self):
#         return "vae"
#
#     def fit(self, training=False):
#         if not training:
#             input_dim = self.dataset.get_train_data().shape[1]
#             self.model = VAEModule(input_dim, 512, 256, 128, 64).to(get_device())
#             self.model.load_state_dict(torch.load(self._checkpoint_path(), map_location=get_device()))
#
#     def predict(self, x: csr_matrix, **kwargs):
#         self.model.eval()
#
#         with torch.no_grad():
#             input_data = torch.tensor(x.toarray(), dtype=torch.float32)
#             recon_batch, mu, log_var = self.model(input_data)
#             predictions = recon_batch.cpu().detach().numpy()
#             predictions[x.nonzero()] = 0
#
#             return predictions
