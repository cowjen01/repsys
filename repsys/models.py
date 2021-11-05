import logging
from typing import Any, Text, Dict
from numpy.typing import NDArray

from .splitter import Splitter

logger = logging.getLogger(__name__)


class Model:
    def __init__(self) -> None:
        self._model = None
        self._splitter = Splitter()

    def name(self) -> Text:
        raise NotImplementedError("You must implement the `name` method")

    def fit(self) -> None:
        raise NotImplementedError("You must implement the `fit` method")

    def compile(self) -> None:
        raise NotImplementedError("You must implement the `compile` method")

    def predict(self, X) -> NDArray:
        raise NotImplementedError("You must implement the `predict` method")

    def save_model(self) -> None:
        """Save a trained model into the file system before the server shuts down"""

    def load_model(self) -> None:
        """Load a trained model from the file system after the server starts up"""

    def prediction_params() -> Dict[Text, Dict[Text, Any]]:
        """Define custom parameters used during the prediction process"""

    def __str__(self) -> Text:
        return f"Model '{self.name()}'"
