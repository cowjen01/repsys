class WebParamType:
    text = "text"
    number = "number"
    select = "select"
    bool = "bool"


class WebParam:
    def __init__(
        self,
        name,
        type=WebParamType.text,
        default_value="",
        label="",
        select_options=[],
    ) -> None:
        self.name = name
        self.label = label
        self.type = type
        self.select_options = select_options
        self.default_value = default_value

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "options": self.select_options,
            "default": self.default_value,
        }
