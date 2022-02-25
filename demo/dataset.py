import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def name(self):
        return "ml20m"

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
        df = pd.read_json("./datasets/ml-20m/items.json")
        df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        return pd.read_csv("./datasets/ml-20m/ratings.csv")


# class TempleWebster(Dataset):
#     def name(self):
#         return "tw"
#
#     def item_cols(self):
#         return {
#             "itemid": dtypes.ItemID(),
#             "title": dtypes.Title(),
#             "description": dtypes.String(),
#             "image_link": dtypes.String(),
#             "availability": dtypes.Category(),
#             "condition": dtypes.Category(),
#             "brand": dtypes.Category()
#         }
#
#     def interaction_cols(self):
#         return {
#             "itemid": dtypes.ItemID(),
#             "userid": dtypes.UserID(),
#             "amount": dtypes.Interaction(),
#         }
#
#     def load_items(self):
#         df = pd.read_csv("./datasets/tw/TW_items.csv")
#         df = df.drop_duplicates(subset=['itemid'])
#         return df
#
#     def load_interactions(self):
#         df = pd.read_csv("./datasets/tw/TW_purchases.csv")
#         df = df[['userid', 'itemid', 'amount']]
#         df = df.groupby(by=['userid', 'itemid'], as_index=False).sum()
#         return df
#


