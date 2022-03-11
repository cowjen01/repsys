from typing import Dict, List, Type, Optional, Tuple, Any


class DataType:
    pass


class Tag(DataType):
    def __init__(self, sep: str = ",") -> None:
        self.sep = sep

    def __str__(self):
        return "tag"


class Category(DataType):
    def __str__(self):
        return "category"


class String(DataType):
    def __str__(self):
        return "string"


class Title(DataType):
    def __str__(self):
        return "title"


class Number(DataType):
    def __init__(
        self,
        data_type: Any = float,
        empty_value: int = 0,
        bins_range: Optional[Tuple[int, int]] = None,
    ):
        self.bins_range = bins_range
        self.empty_value = empty_value
        self.data_type = data_type

    def __str__(self):
        return "number"


class UserID(DataType):
    def __str__(self):
        return "id"


class ItemID(DataType):
    def __str__(self):
        return "id"


class Interaction(DataType):
    def __str__(self):
        return "interaction"


ColumnDict = Dict[str, Type[DataType]]


def filter_columns_by_type(columns: ColumnDict, dtype: Type[DataType]) -> List[str]:
    results = []
    for [col, dt] in columns.items():
        if type(dt) == dtype:
            results.append(col)

    return results


def find_column_by_type(columns: ColumnDict, dtype: Type[DataType]) -> Optional[str]:
    cols = filter_columns_by_type(columns, dtype)
    return cols[0] if cols else None
