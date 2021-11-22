class WebsiteParam:
    """Optional parameter passed to the prediction method."""

    def __init__(
        self, key, label, select_options=[], type="text", default_value=""
    ) -> None:
        self.key = key
        self.label = label
        self.type = type
        self.select_options = select_options
        self.default_value = default_value
