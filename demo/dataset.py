import pandas as pd

from repsys import Dataset
import repsys.dataset.dtypes as dtypes


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return (
            self.items["genres"]
            .dropna()
            .str.split("|", expand=True)
            .stack()
            .unique()
        )

    def item_dtypes(self):
        return {
            "movieId": dtypes.ItemIndex(),
            "title": dtypes.String(),
            "genres": dtypes.Tags(sep="|"),
        }

    def interact_dtypes(self):
        return {
            "movieId": dtypes.ItemIndex(),
            "userId": dtypes.UserIndex(),
            "rating": dtypes.Rating(step=0.5),
        }

    def load_items(self):
        return pd.read_csv("./ml-latest-small/movies.csv")

    def load_interacts(self):
        return pd.read_csv("./ml-latest-small/ratings.csv")
