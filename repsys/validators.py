from typing import List, Type

from pandas import DataFrame

from repsys.dtypes import (
    DataType,
    ColumnDict,
    Number,
    String,
    UserID,
    ItemID,
    Category,
    Interaction,
    Tag,
    Title,
    find_column_by_type,
)
from repsys.errors import InvalidDatasetError


def _check_df_columns(df: DataFrame, cols: ColumnDict):
    for col in cols.keys():
        if col not in df.columns:
            raise InvalidDatasetError(f"Column '{col}' not found in the data.")


def _check_valid_dtypes(cols: ColumnDict, valid_dtypes: List[Type[DataType]]):
    for col, dt in cols.items():
        if type(dt) not in valid_dtypes:
            raise InvalidDatasetError(f"Type '{dt.__name__}' of column '{col}' is forbidden.")


def _check_required_dtypes(cols: ColumnDict, req_dtypes: List[Type[DataType]]):
    dtypes = [type(dt) for dt in cols.values()]
    for dt in req_dtypes:
        if dt not in dtypes:
            raise InvalidDatasetError(f"Type '{dt.__name__}' is required.")


def validate_item_cols(cols: ColumnDict) -> None:
    valid_dtypes = [ItemID, Tag, String, Title, Number, Category]
    required_dtypes = [ItemID, Title]

    _check_valid_dtypes(cols, valid_dtypes)
    _check_required_dtypes(cols, required_dtypes)


def validate_item_data(items: DataFrame, cols: ColumnDict) -> None:
    _check_df_columns(items, cols)

    item_col = find_column_by_type(cols, ItemID)
    if items.duplicated(subset=[item_col]).sum() > 0:
        raise InvalidDatasetError(f"Index '{item_col}' contains non-unique values.")


def validate_interact_cols(cols: ColumnDict) -> None:
    valid_dtypes = [ItemID, UserID, Interaction]
    required_dtypes = [ItemID, UserID]

    _check_valid_dtypes(cols, valid_dtypes)
    _check_required_dtypes(cols, required_dtypes)


def validate_interact_data(
    interacts: DataFrame,
    items: DataFrame,
    interact_cols: ColumnDict,
    item_cols: ColumnDict,
) -> None:
    _check_df_columns(interacts, interact_cols)

    interacts_item_id_col = find_column_by_type(interact_cols, ItemID)
    items_id_col = find_column_by_type(item_cols, ItemID)

    s1 = set(interacts[interacts_item_id_col])
    s2 = set(items[items_id_col])

    diff = s1.difference(s2)

    if len(diff) > 0:
        raise InvalidDatasetError(
            "Some of the items are included in the interactions data " f"but not in the items data: {list(diff)}."
        )


def validate_dataset(
    items: DataFrame,
    item_cols: ColumnDict,
    interacts: DataFrame,
    interact_cols: ColumnDict,
):
    validate_item_cols(item_cols)
    validate_item_data(items, item_cols)
    validate_interact_cols(interact_cols)
    validate_interact_data(interacts, items, interact_cols, item_cols)
