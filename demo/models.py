import numpy as np
from sklearn.neighbors import NearestNeighbors
from repsys import Model, WebsiteParam

# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab


class KNN(Model):
    def __init__(self, k=10):
        super().__init__()
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def name(self):
        return "KNN10"

    def fit(self):
        self.model.fit(self.dataset.train_data)

    def website_params(self):
        return [
            WebsiteParam(
                key="exclude_history",
                label="Exclude user's history",
                type="boolean",
                default_value=True,
            ),
            WebsiteParam(
                key="normalize_distances",
                label="Normalize neighbors distances",
                type="boolean",
                default_value=True,
            )
        ]

    def predict(self, X, **kwargs):
        distances, indexes = self.model.kneighbors(X)

        # exclude the nearest neighbor
        n_distances = distances[:, 1:]
        n_indexes = indexes[:, 1:]

        # invert the distance into weight
        n_distances = 1 - n_distances

        if kwargs["normalize_distances"]:
            # normalize the distances
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

        if kwargs["exclude_history"]:
            # remove items user interacted with
            predictions[X.toarray() > 0] = 0

        return predictions


class KNN5(KNN):
    def __init__(self):
        super().__init__(k=5)

    def name(self):
        return "KNN5"
