import pytest

from .helpers import load_items, load_interacts

from repsys import dtypes
from repsys.validators import (
    validate_dataset,
    validate_item_data,
    validate_item_dtypes,
    validate_interact_dtypes,
    validate_interact_data,
)


def get_valid_setup():
    items = load_items("valid.csv")
    interacts = load_interacts("valid.csv")
    interact_dt = {
        "movieId": dtypes.ItemID(),
        "userId": dtypes.UserID(),
        "rating": dtypes.Rating(min=0.5, max=5, step=0.5),
    }
    item_dt = {
        "movieId": dtypes.ItemID(),
        "title": dtypes.Title(),
        "genres": dtypes.Tags(),
    }

    return interacts, items, interact_dt, item_dt


@pytest.mark.parametrize(
    "dt",
    [
        {
            "movieId": dtypes.ItemID(),
            "title2": dtypes.Title(),
        },
        {
            "movieId": dtypes.String(),
            "title": dtypes.Title(),
        },
        {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
        },
    ],
)
def test_invalid_item_dtypes(dt):
    items = load_items("valid.csv")
    with pytest.raises(Exception):
        validate_item_dtypes(items, dt)


@pytest.mark.parametrize(
    "dt",
    [
        {
            "rating": dtypes.Rating(step=0.5),
        },
        {
            "movieId": dtypes.ItemID(),
        },
        {
            "userId": dtypes.UserID(),
        },
        {
            "movieId": dtypes.ItemID(),
            "userId": dtypes.UserID(),
            "genres": dtypes.Tags(),
        },
        {
            "movieIndex": dtypes.ItemID(),
            "userId": dtypes.UserID(),
        },
    ],
)
def test_invalid_interact_dtypes(dt):
    interacts = load_interacts("valid.csv")
    with pytest.raises(Exception):
        validate_interact_dtypes(interacts, dt)


@pytest.mark.parametrize("items", [load_items("duplicate_index.csv")])
def test_invalid_item_data(items):
    dt = {
        "movieId": dtypes.ItemID(),
        "title": dtypes.Title(),
    }
    with pytest.raises(Exception):
        validate_item_data(items, dt)


@pytest.mark.parametrize(
    "interacts",
    [load_interacts("invalid_rating.csv"), load_interacts("uknown_items.csv")],
)
def test_invalid_interact_data(interacts):
    _, items, interact_dt, item_dt = get_valid_setup()
    with pytest.raises(Exception):
        validate_interact_data(interacts, items, interact_dt, item_dt)


def test_valid_dataset():
    interacts, items, interact_dt, item_dt = get_valid_setup()
    validate_dataset(interacts, items, interact_dt, item_dt)
