import pandas as pd

from repsys.dataset import Dataset
import repsys.dataset.dtypes as dtypes


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return self.tags['genres']

    def get_item_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.String(),
            "genres": dtypes.Tags(sep="|"),
        }

    def get_interact_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Rating(min=0.5, step=0.5),
        }

    def load_items(self):
        return pd.read_csv("./ml-20m/movies.csv")

    def load_interacts(self):
        return pd.read_csv("./ml-20m/ratings.csv")
