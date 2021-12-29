import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

from repsys import Model
from repsys.web import Select

# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/


class BaseKNN(Model):
    def __init__(self, k=15):
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.name())

    def _serialize(self):
        return self.model

    def _unserialize(self, checkpoint):
        self.model = checkpoint

    def _load_model(self):
        checkpoint = pickle.load(open(self._checkpoint_path(), "rb"))
        self._unserialize(checkpoint)

    def _save_model(self):
        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self._serialize(), checkpoint)

    def fit(self, training):
        if training:
            self.model.fit(self.dataset.train_data)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X, **kwargs):
        distances, indexes = self.model.kneighbors(X)

        # exclude the nearest neighbor
        n_distances = distances[:, 1:]
        n_indexes = indexes[:, 1:]

        # invert the distance into weight
        n_distances = 1 - n_distances

        # normalize distances
        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        # 1) get interactions of the nearest neighbors
        # 2) multiply them by a distance from the user
        # 3) sum up the weighted interactions of the nearest users
        # 4) turn the numpy matrices into a numpy array and squeeze
        predictions = np.array(
            [
                self.dataset.train_data[idx]
                .multiply(dist.reshape(-1, 1))
                .sum(axis=0)
                for idx, dist in zip(n_indexes, n_distances)
            ]
        ).squeeze(axis=1)

        if kwargs.get("movie_genre"):
            genre = kwargs.get("movie_genre")

            # exclude movies without the genre
            excluded_ids = self.dataset.items.loc[
                ~self.dataset.items["genres"].str.contains(genre)
            ].index
            excluded_idxs = excluded_ids.map(self.dataset.get_item_index)
            predictions[:, excluded_idxs] = 0

        return predictions

    def web_params(self):
        return [
            Select(
                "movie_genre",
                options=self.dataset.get_genres(),
                label="Movie genre",
            )
        ]


class KNN(BaseKNN):
    def name(self):
        return "KNN"

    def predict(self, X, **kwargs):
        predictions = super().predict(X, **kwargs)
        predictions[X.toarray() > 0] = 0
        return predictions


class SVDKNN(BaseKNN):
    def __init__(self, svd_components=50):
        super().__init__()
        self.svd = TruncatedSVD(n_components=svd_components, algorithm="arpack")

    def name(self):
        return "SVDKNN"

    def fit(self, training):
        if training:
            train_embed = self.svd.fit_transform(self.dataset.train_data)
            self.model.fit(train_embed)
            self._save_model()
        else:
            self._load_model()

    def predict(self, X, **kwargs):
        X_embed = self.svd.transform(X)
        predictions = super().predict(X_embed, **kwargs)
        predictions[X.toarray() > 0] = 0
        return predictions

    def _serialize(self):
        return {"knn": self.model, "svd": self.svd}

    def _unserialize(self, checkpoint):
        self.model = checkpoint.get("knn")
        self.svd = checkpoint.get("svd")
