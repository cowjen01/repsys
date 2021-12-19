from typing import Text, Dict, Optional


class DataType:
    pass


class Tags(DataType):
    def __init__(self, sep: Text = ",") -> None:
        self.sep = sep


class String(DataType):
    pass


class UserIndex(DataType):
    pass


class ItemIndex(DataType):
    pass


class ExplicitInteraction(DataType):
    pass


class Rating(ExplicitInteraction):
    def __init__(
        self,
        min: float = 1.0,
        max: float = 5.0,
        step: float = 1.0,
        bin_threshold: float = 4.0,
    ) -> None:
        self.min = min
        self.max = max
        self.step = step
        self.bin_threshold = bin_threshold


def find_column(
    dtypes: Dict[Text, DataType], col_dtype: DataType
) -> Optional[Text]:
    for [col, dt] in dtypes.items():
        if type(dt) == col_dtype:
            return col
    return None


def filter_columns(
    dtypes: Dict[Text, DataType], col_dtype: DataType
) -> Optional[Text]:
    results = []
    for [col, dt] in dtypes.items():
        if type(dt) == col_dtype:
            results.append(col)
    return results
