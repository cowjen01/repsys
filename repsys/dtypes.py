from typing import Dict, List, Type, Optional


class DataType:
    pass


class Tags(DataType):
    def __init__(self, sep: str = ",") -> None:
        self.sep = sep


class Category(DataType):
    pass


class String(DataType):
    pass


class Title(DataType):
    pass


class Number(DataType):
    pass


class UserID(DataType):
    pass


class ItemID(DataType):
    pass


class Interaction(DataType):
    pass


def filter_columns_by_type(
    columns: Dict[str, Type[DataType]], dtype: Type[DataType]
) -> List[str]:
    results = []
    for [col, dt] in columns.items():
        if type(dt) == dtype:
            results.append(col)

    return results


def find_column_by_type(columns: Dict[str, Type[DataType]],
                        dtype: Type[DataType]) -> Optional[str]:
    cols = filter_columns_by_type(columns, dtype)
    return cols[0] if cols else None
