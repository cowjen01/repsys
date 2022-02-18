class InvalidConfigError(Exception):
    def __init__(self, message):
        self.message = f'Reading of the config file failed: {message}'
        super().__init__(self.message)


class ApiHttpError(Exception):
    def __init__(self, message, code=500):
        self.code = code
        self.message = f'The request to the repository failed: {message}'
        super().__init__(self.message)


class RepsysCoreError(Exception):
    def __init__(self, message):
        self.message = f'Executing of the command failed: {message}'
        super().__init__(self.message)
