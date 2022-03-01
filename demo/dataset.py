import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


# class MovieLens(Dataset):
#     def __init__(self):
#         super(MovieLens, self).__init__()
#         self._nmf = NMF(n_components=10, init='nndsvd', max_iter=100, random_state=0, verbose=2)
#
#     def name(self):
#         return "ml20m"
#
#     def compute_embeddings(self, X: csr_matrix):
#         W = self._nmf.fit_transform(X)
#         H = self._nmf.components_
#         return W, H.T
#
#     def item_cols(self):
#         return {
#             "itemid": dtypes.ItemID(),
#             "product_name": dtypes.Title(),
#             "description": dtypes.String(),
#             "image": dtypes.String(),
#             "director": dtypes.String(),
#             "language": dtypes.Tag(sep=", "),
#             "genre": dtypes.Tag(sep=", "),
#             "country": dtypes.Tag(sep=", "),
#             "year": dtypes.Number(data_type=int),
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
#         df = pd.read_json("./datasets/ml-20m/items.json")
#         df["year"] = df["product_name"].str.extract(r"\((\d+)\)")
#         return df
#
#     def load_interactions(self):
#         df = pd.read_csv("./datasets/ml-20m/ratings.csv")
#         df = df[df['rating'] > 3.5]
#         df.loc[df['rating'] > 0, 'rating'] = 1
#         return df

class TempleWebster(Dataset):
    def name(self):
        return "tw"

    def item_cols(self):
        return {
            "itemid": dtypes.ItemID(),
            "title": dtypes.Title(),
            "description": dtypes.String(),
            "image_link": dtypes.String(),
            "material": dtypes.Tag(sep=','),
            "product_type": dtypes.Category(),
            "price": dtypes.Number(),
            "brand": dtypes.Category()
        }

    def interaction_cols(self):
        return {
            "itemid": dtypes.ItemID(),
            "userid": dtypes.UserID(),
            "amount": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_csv("./datasets/tw/TW_items.csv")
        df = df.drop_duplicates(subset=['itemid'])
        df['product_type'] = df['product_type'].fillna('')
        df['product_type'] = df['product_type'].apply(lambda x: x[2:-2])
        return df

    def load_interactions(self):
        df = pd.read_csv("./datasets/tw/TW_purchases.csv")
        df = df[['userid', 'itemid', 'amount']]
        df = df.groupby(by=['userid', 'itemid'], as_index=False).median()
        # df.loc[df['amount'] > 0, 'amount'] = 1
        return df
