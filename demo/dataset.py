import pandas as pd

import repsys.dtypes as dtypes
from repsys.dataset import Dataset


class MovieLens(Dataset):
    def name(self):
        return "ml20m"

    def item_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.Title(),
            "genres": dtypes.Tag(sep="|"),
            "year": dtypes.Number(data_type=int),
        }

    def interaction_cols(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "rating": dtypes.Interaction(),
        }

    def load_items(self):
        df = pd.read_csv("./ml-20m/movies.csv")
        df["year"] = df["title"].str.extract(r"\((\d+)\)")
        return df

    def load_interactions(self):
        df = pd.read_csv("./ml-20m/ratings.csv")
        df = df[df["rating"] > 3.5]
        df["rating"] = 1
        return df


# class BookCrossing(Dataset):
#     def __init__(self) -> None:
#         self.books = pd.read_csv("./bx-csv/BX-Books.csv", sep=";", escapechar="\\", encoding="CP1252")

#     def name(self):
#         return "bx"

#     def item_cols(self):
#         return {
#             "ISBN": dtypes.ItemID(),
#             "Book-Title": dtypes.Title(),
#             "Book-Author": dtypes.Category(),
#             "Year-Of-Publication": dtypes.Number(data_type=int),
#             "Publisher": dtypes.Category(),
#             "Image-URL-L": dtypes.String()
#         }

#     def interaction_cols(self):
#         return {
#             "ISBN": dtypes.ItemID(),
#             "User-ID": dtypes.UserID(),
#         }

#     def load_items(self):
#         return self.books

#     def load_interactions(self):
#         df = pd.read_csv("./bx-csv/BX-Book-Ratings.csv", sep=";", escapechar="\\", encoding="CP1252")

#         df = df[df["ISBN"].isin(self.books["ISBN"].unique())]
#         df = df[df["Book-Rating"] >= 6]

#         return df
