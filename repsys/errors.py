class InvalidConfigError(Exception):
    def __init__(self, message):
        self.message = f"Reading of the config file failed: {message}"
        super().__init__(self.message)
