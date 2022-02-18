import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return self.tags.get('genres')

    def item_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.Title(),
            "genres": dtypes.Tag(sep="|"),
            "year": dtypes.Number(),
        }

    def interaction_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_csv("./ml-sm/movies.csv")
        df["year"] = df["title"].str.extract("\((\d+)\)")
        df["year"] = df["year"].fillna(0)
        df["year"] = df["year"].astype(int)
        return df

    def load_interactions(self):
        return pd.read_csv("./ml-sm/ratings.csv")
