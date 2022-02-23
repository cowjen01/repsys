import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return self.tags.get('genre')

    def item_cols(self):
        return {
            "itemid": dtypes.ItemID(),
            "product_name": dtypes.Title(),
            "description": dtypes.String(),
            "image": dtypes.String(),
            "director": dtypes.String(),
            "language": dtypes.Tag(sep=", "),
            "genre": dtypes.Tag(sep=", "),
            "country": dtypes.Tag(sep=", "),
            "year": dtypes.Number(data_type=int),
        }

    def interaction_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_json("./ml-20m/items.json")
        df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        return pd.read_csv("./ml-20m/ratings.csv")

#
# class MovieLens(Dataset):
#     def name(self):
#         return "movielens"
#
#     def get_genres(self):
#         return self.tags.get('genres')
#
#     def item_cols(self):
#         return {
#             "movieId": dtypes.ItemID(),
#             "title": dtypes.Title(),
#             "genres": dtypes.Tag(sep="|"),
#             "year": dtypes.Number(data_type=int)
#         }
#
#     def interaction_cols(self):
#         return {
#             "movieId": dtypes.ItemID(),
#             "userId": dtypes.UserID(),
#             "rating": dtypes.Interaction(),
#         }
#
#     def load_items(self):
#         df = pd.read_csv("./ml-sm/movies.csv")
#         df["year"] = df["title"].str.extract(r"\((\d+)\)")
#         return df
#
#     def load_interactions(self):
#         return pd.read_csv("./ml-sm/ratings.csv")
