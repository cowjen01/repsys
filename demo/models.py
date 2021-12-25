import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# from sklearn.decomposition import TruncatedSVD
from repsys import Model
from repsys.web import Select

# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/


# def erase_history(f):
#     def w(self, X, **kwargs):
#         p = f(self, X, **kwargs)
#         p[X.toarray() > 0] = 0
#         return p

#     return w


class KNN(Model):
    def __init__(self, k=5):
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def name(self):
        return "KNN"

    def _checkpoint_path(self):
        return os.path.join("./checkpoints", self.name())

    def _load_model(self):
        self.model = pickle.load(open(self._checkpoint_path(), "rb"))

    def _save_model(self):
        checkpoint = open(self._checkpoint_path(), "wb")
        pickle.dump(self.model, checkpoint)

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

        predictions[X.toarray() > 0] = 0

        if kwargs.get("movie_genre"):
            # exclude movies without the genre
            genre_mask = (
                ~self.dataset.items["genres"]
                .str.contains(kwargs["movie_genre"])
                .sort_index()
            )
            predictions[:, genre_mask] = 0

        return predictions

    def web_params(self):
        return [
            Select(
                "movie_genre",
                options=self.dataset.get_genres(),
                label="Movie genre",
            )
        ]


# class BaseKNN(KNN):
#     def name(self):
#         return "BaseKNN"

#     def fit(self):
#         self.model.fit(self.dataset.train_data)

#     @erase_history
#     def predict(self, X, **kwargs):
#         return super().predict(X, **kwargs)


# class SVDKNN(KNN):
#     def __init__(self, svd_components=20):
#         super().__init__()
#         self.svd = TruncatedSVD(n_components=svd_components, algorithm="arpack")

#     def name(self):
#         return "SVDKNN"

#     def fit(self):
#         self.train_embed = self.svd.fit_transform(self.dataset.train_data)
#         self.model.fit(self.train_embed)

#     @erase_history
#     def predict(self, X, **kwargs):
#         X_embed = self.svd.transform(X)
#         return super().predict(X_embed, **kwargs)

#     def serialize(self):
#         return {"knn": self.model, "svd": self.svd}

#     def unserialize(self, state):
#         self.model = state.get("knn")
#         self.svd = state.get("svd")
