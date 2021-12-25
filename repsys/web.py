class WebParam:
    def __init__(
        self,
        type,
        name,
        default="",
        label="",
    ) -> None:
        self.name = name
        self.label = label
        self.type = type
        self.default = default

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "default": self.default,
        }


class Text(WebParam):
    def __init__(self, name, default="", label="") -> None:
        super().__init__("text", name, default, label)


class Select(WebParam):
    def __init__(self, name, options, default="", label="") -> None:
        super().__init__("select", name, default, label)
        self.options = options

    def to_dict(self):
        data = super().to_dict()
        data["options"] = self.options
        return data


class Boolean(WebParam):
    def __init__(self, name, default=False, label="") -> None:
        super().__init__("bool", name, default, label)


class Number(WebParam):
    def __init__(self, name, default="", label="") -> None:
        super().__init__("number", name, default, label)
