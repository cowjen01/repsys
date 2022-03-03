import pytest

from repsys.dtypes import (
    filter_columns,
    find_column,
    ItemID,
    UserID,
    Tags,
    String,
)


@pytest.mark.parametrize(
    "dt,expected",
    [(UserID, "userId"), (Tags, "genres"), (String, None)],
)
def test_column_find(dt, expected):
    dts = {
        "movieId": ItemID(),
        "userId": UserID(),
        "genres": Tags(),
    }

    assert find_column(dts, dt) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [(Tags, ["genres", "languages"]), (String, ["title"]), (UserID, [])],
)
def test_columns_filter(dt, expected):
    dts = {
        "movieId": ItemID(),
        "genres": Tags(),
        "languages": Tags(),
        "title": String(),
    }

    assert filter_columns(dts, dt) == expected
