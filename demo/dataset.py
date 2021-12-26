import pandas as pd

from repsys.dataset import Dataset
import repsys.dtypes as dtypes


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return self.tags["genres"]

    def get_item_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.Title(),
            "genres": dtypes.Tags(sep="|"),
            "year": dtypes.String(),
        }

    def get_interact_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Rating(min=0.5, step=0.5),
        }

    def load_items(self):
        df = pd.read_csv("./ml-sm/movies.csv")
        df["year"] = df["title"].str.extract(r"\((\d+)\)")
        return df

    def load_interacts(self):
        return pd.read_csv("./ml-sm/ratings.csv")
