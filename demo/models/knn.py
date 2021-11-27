import numpy as np
from sklearn.neighbors import NearestNeighbors
from repsys import ScikitModel, PredictParam, ParamTypes

# https://gist.github.com/mskl/fcc3c432e00e417cec670c6c3a45d6ab
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/


class KNN(ScikitModel):
    def __init__(self, k=5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine")

    def name(self):
        return "KNN5"

    def fit(self):
        self.model.fit(self.dataset.train_data)

    def predict(self, X, **kwargs):
        distances, indexes = self.model.kneighbors(X, n_neighbors=self.k)

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

        # remove items user interacted with
        predictions[X.toarray() > 0] = 0

        if kwargs["movie_genre"]:
            # exclude movies without the genre
            genre_mask = (
                ~self.dataset.items["subtitle"]
                .str.contains(kwargs["movie_genre"])
                .sort_index()
            )
            predictions[:, genre_mask] = 0

        return predictions

    def predict_params(self):
        return [
            PredictParam(
                name="movie_genre",
                label="Movie genre",
                type=ParamTypes.select,
                select_options=self.dataset.get_genres().tolist(),
            ),
        ]


class KNN5(KNN):
    def __init__(self):
        super().__init__(k=10)

    def name(self):
        return "KNN10"
