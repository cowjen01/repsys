class WebParam:
    def __init__(
        self,
        field,
        default="",
    ) -> None:
        self.field = field
        self.default = default

    def to_dict(self):
        return {
            "field": self.field,
            "default": self.default,
        }


class Text(WebParam):
    def __init__(self, default="") -> None:
        super().__init__("text", default)


class Select(WebParam):
    def __init__(self, options, default="") -> None:
        super().__init__("select", default)
        self.options = options

    def to_dict(self):
        data = super().to_dict()
        data["options"] = self.options
        return data


class Checkbox(WebParam):
    def __init__(self, default=False) -> None:
        super().__init__("checkbox", default)


class Number(WebParam):
    def __init__(self, default="") -> None:
        super().__init__("number", default)
