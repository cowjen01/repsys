import pytest

from tests.helpers import load_items, load_interacts

from repsys.dataset import dtypes
from repsys.dataset.validation import (
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
        "movieId": dtypes.ItemIndex(),
        "userId": dtypes.UserIndex(),
        "rating": dtypes.Rating(min=0.5, max=5, step=0.5),
    }
    item_dt = {
        "movieId": dtypes.ItemIndex(),
        "title": dtypes.String(),
        "genres": dtypes.Tags(),
    }

    return interacts, items, interact_dt, item_dt


@pytest.mark.parametrize(
    "dt",
    [
        {
            "movieId": dtypes.ItemIndex(),
            "title2": dtypes.String(),
        },
        {
            "movieId": dtypes.String(),
            "title": dtypes.String(),
        },
        {
            "movieId": dtypes.ItemIndex(),
            "userId": dtypes.UserIndex(),
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
            "movieId": dtypes.ItemIndex(),
        },
        {
            "userId": dtypes.UserIndex(),
        },
        {
            "movieId": dtypes.ItemIndex(),
            "userId": dtypes.UserIndex(),
            "genres": dtypes.Tags(),
        },
        {
            "movieIndex": dtypes.ItemIndex(),
            "userId": dtypes.UserIndex(),
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
        "movieId": dtypes.ItemIndex(),
        "title": dtypes.String(),
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
