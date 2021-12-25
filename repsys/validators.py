from typing import Dict, Text, List, Optional
from pandas import DataFrame
import numpy as np
import logging

from repsys.dtypes import (
    DataType,
    Rating,
    String,
    UserID,
    ItemID,
    Tags,
    find_column,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    def __init__(col_name=None, dtype_name=None):
        col_name = col_name
        dtype_name = dtype_name


# FIXME TypeError: __init__() got multiple values for argument 'col_name
def check_columns(columns: List[Text], df_columns: List[Text]):
    for col in columns:
        if col not in df_columns:
            raise ValidationError(col_name=col)


def check_valid_dtypes(dtypes: List[DataType], valid_dtypes: List[DataType]):
    for dt in dtypes:
        if type(dt) not in valid_dtypes:
            raise ValidationError(dtype_name=type(dt).__name__)


def check_required_dtypes(dtypes: List[DataType], req_dtypes: List[DataType]):
    dtypes = [type(dt) for dt in dtypes]
    for dt in req_dtypes:
        if dt not in dtypes:
            raise ValidationError(dtype_name=dt.__name__)


def validate_item_dtypes(
    items: DataFrame, item_dtypes: Dict[Text, DataType]
) -> None:
    valid_dtypes = [ItemID, Tags, String]
    required_dtypes = [ItemID]

    try:
        check_columns(item_dtypes.keys(), items.columns)
    except ValidationError as e:
        raise Exception(f"Column '{e.col_name}' not found in the items data.")

    try:
        check_valid_dtypes(item_dtypes.values(), valid_dtypes)
    except ValidationError as e:
        raise Exception(f"Type '{e.dtype_name}' is forbidden for item dtypes.")

    try:
        check_required_dtypes(item_dtypes.values(), required_dtypes)
    except ValidationError as e:
        raise Exception(f"Type '{e.dtype_name}' is required for item dtypes.")


def validate_item_data(
    items: DataFrame, item_dtypes: Dict[Text, DataType]
) -> None:
    item_index = find_column(item_dtypes, ItemID)
    if items[item_index].unique().shape[0] != items.shape[0]:
        raise Exception("Index '{item_index}' contains non-unique values.")


def validate_interact_dtypes(
    interacts: DataFrame, interact_dtypes: Dict[Text, DataType]
) -> None:
    valid_dtypes = [ItemID, UserID, Rating]
    required_dtypes = [ItemID, UserID]

    try:
        check_columns(interact_dtypes.keys(), interacts.columns)
    except ValidationError as e:
        raise Exception(
            f"Column '{e.col_name}' not found in the interactions data."
        )

    try:
        check_valid_dtypes(interact_dtypes.values(), valid_dtypes)
    except ValidationError as e:
        raise Exception(
            f"Type '{e.dtype_name}' is forbidden for interaction dtypes."
        )

    try:
        check_required_dtypes(interact_dtypes.values(), required_dtypes)
    except ValidationError as e:
        raise Exception(
            f"Type '{e.dtype_name}' is required for interaction dtypes."
        )


def validate_interact_data(
    interacts: DataFrame,
    items: DataFrame,
    interact_dtypes: Dict[Text, DataType],
    item_dtypes: Dict[Text, DataType],
) -> None:
    interacts_item_idx = find_column(interact_dtypes, ItemID)
    items_item_idx = find_column(item_dtypes, ItemID)

    s1 = set(interacts[interacts_item_idx])
    s2 = set(items[items_item_idx])

    diff = s1.difference(s2)

    if len(diff) > 0:
        raise Exception(
            "Some items are included in the interactions data "
            f"but not in the items data: {list(diff)}."
        )

    rating_index = find_column(interact_dtypes, Rating)

    if rating_index:
        dt: Rating = interact_dtypes[rating_index]

        s1 = set(interacts[rating_index].values)
        s2 = set(np.arange(dt.min, dt.max + dt.step, dt.step))

        diff = s1.difference(s2)

        if len(diff) > 0:
            raise Exception(
                "The rating column does not match specified constraints "
                f"(min: {dt.min}, max: {dt.max}, step: {dt.step})."
            )


def validate_item_view(
    item_view: Dict[Text, Optional[Text]], item_dtypes: Dict[Text, DataType]
):
    if not item_view.get("title"):
        raise Exception(
            "Item's title column was not recognized. Please specify "
            "item view implementing `item_view()` method."
        )

    for col in item_view.values():
        if col is not None and col not in item_dtypes.keys():
            raise Exception(
                f"Item's view column '{col}' was not found "
                "in the item's dtypes specification."
            )


def validate_dataset(
    interacts: DataFrame,
    items: DataFrame,
    interact_dtypes: Dict[Text, DataType],
    item_dtypes: Dict[Text, DataType],
):
    validate_item_dtypes(items, item_dtypes)
    validate_item_data(items, item_dtypes)
    validate_interact_dtypes(interacts, interact_dtypes)
    validate_interact_data(interacts, items, interact_dtypes, item_dtypes)
