import pytest

from repsys import dtypes

from .helpers import load_items, load_interacts

from repsys.dataset import Dataset


class TestDataset(Dataset):
    def name(self):
        return "testdataset"

    def item_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.String(),
            "genres": dtypes.Tags(sep="|"),
        }

    def interact_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
        }

    def load_items(self):
        return load_items("valid.csv")

    def load_interacts(self):
        return load_interacts("valid.csv")
