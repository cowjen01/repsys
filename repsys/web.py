class WebParam:
    def __init__(
        self,
        field,
        name,
        default="",
    ) -> None:
        self.name = name
        self.field = field
        self.default = default

    def to_dict(self):
        return {
            "name": self.name,
            "field": self.field,
            "default": self.default,
        }


class Text(WebParam):
    def __init__(self, name, default="") -> None:
        super().__init__("text", name, default)


class Select(WebParam):
    def __init__(self, name, options, default="") -> None:
        super().__init__("select", name, default)
        self.options = options

    def to_dict(self):
        data = super().to_dict()
        data["options"] = self.options
        return data


class Boolean(WebParam):
    def __init__(self, name, default=False) -> None:
        super().__init__("bool", name, default)


class Number(WebParam):
    def __init__(self, name, default="") -> None:
        super().__init__("number", name, default)
