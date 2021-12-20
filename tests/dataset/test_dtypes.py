import pytest

from repsys.dataset.dtypes import (
    filter_columns,
    find_column,
    ItemIndex,
    UserIndex,
    Tags,
    String,
)


@pytest.mark.parametrize(
    "dt,expected",
    [(UserIndex, "userId"), (Tags, "genres"), (String, None)],
)
def test_column_find(dt, expected):
    dts = {
        "movieId": ItemIndex(),
        "userId": UserIndex(),
        "genres": Tags(),
    }

    assert find_column(dts, dt) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [(Tags, ["genres", "languages"]), (String, ["title"]), (UserIndex, [])],
)
def test_columns_filter(dt, expected):
    dts = {
        "movieId": ItemIndex(),
        "genres": Tags(),
        "languages": Tags(),
        "title": String(),
    }

    assert filter_columns(dts, dt) == expected
