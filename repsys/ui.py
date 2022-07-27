from typing import List, Dict, Any


class WebParam:
    def __init__(self, field: str, default: Any = "") -> None:
        self.field = field
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "default": self.default,
        }


class Text(WebParam):
    def __init__(self, default: str = "") -> None:
        super().__init__("text", default)


class Select(WebParam):
    def __init__(self, options: List[str], default: str = "") -> None:
        super().__init__("select", default)
        self.options = options

    def to_dict(self):
        data = super().to_dict()
        data["options"] = self.options
        return data


class Checkbox(WebParam):
    def __init__(self, default: bool = False) -> None:
        super().__init__("checkbox", default)


class Number(WebParam):
    def __init__(self, default: Any = "") -> None:
        super().__init__("number", default)
