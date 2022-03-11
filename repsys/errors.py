class InvalidConfigError(Exception):
    def __init__(self, message):
        self.message = f"Reading of the config file failed: {message}"
        super().__init__(self.message)


class RepsysCoreError(Exception):
    def __init__(self, message):
        self.message = f"Executing of the command failed: {message}"
        super().__init__(self.message)


class InvalidDatasetError(Exception):
    def __init__(self, message):
        self.message = f"Dataset validation failed: {message}"
        super().__init__(self.message)


class PackageLoaderError(Exception):
    def __init__(self, message):
        self.message = f"Packages loading failed: {message}"
        super().__init__(self.message)
