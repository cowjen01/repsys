import pytest

from repsys import dtypes

from .helpers import load_items, load_interacts

from repsys.dataset import Dataset


class TestDataset(Dataset):
    def name(self):
        return "testdataset"

    def get_item_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "title": dtypes.String(),
            "genres": dtypes.Tags(sep="|"),
        }

    def get_item_view(self):
        return {"caption": "genres"}

    def get_interact_dtypes(self):
        return {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
        }

    def load_items(self):
        return load_items("valid.csv")

    def load_interacts(self):
        return load_interacts("valid.csv")


def test_item_view_merge():
    dataset = TestDataset()
    items = dataset.load_items()
    view = dataset.get_item_view()
    item_view = dataset._merge_item_views(view, items.columns)

    assert item_view.get('title') == 'title'
    assert item_view.get('caption') == 'genres'
    assert item_view.get('image') is None
