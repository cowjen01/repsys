import numpy as np
from sklearn.neighbors import NearestNeighbors
from repsys import Model

# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab


class KNN(Model):
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def name(self):
        return "KNN10"

    def compile(self):
        self._model = NearestNeighbors(n_neighbors=self.k, metric="cosine")

    def fit(self):
        self._model.fit(self._splitter.train_data)

    def prediction_params():
        return {
            "exclude_history": {
                "type": "boolean",
                "label": "Exclude user's history",
                "default": True,
            }
        }

    def predict(self, X, **kwargs):
        distances, indexes = self._model.kneighbors(X)

        # exclude the nearest neighbor
        n_distances = distances[:, 1:]
        n_indexes = indexes[:, 1:]

        # invert the distance into weight
        n_distances = 1 - n_distances

        # normalize the distances
        sums = n_distances.sum(axis=1)
        n_distances = n_distances / sums[:, np.newaxis]

        # 1) get interactions of the nearest neighbors
        # 2) multiply them by a distance from the user
        # 3) sum up the weighted interactions of the nearest users
        # 4) turn the numpy matrices into a numpy array and squeeze
        predictions = np.array(
            [
                self._splitter.train_data[idx]
                .multiply(dist.reshape(-1, 1))
                .sum(axis=0)
                for idx, dist in zip(n_indexes, n_distances)
            ]
        ).squeeze()

        if kwargs["exclude_history"]:
            # remove items user interacted with
            predictions[X.toarray() > 0] = 0

        return predictions


class KNN5(KNN):
    def __init__(self):
        super().__init__(k=5)

    def name(self):
        return "KNN5"
