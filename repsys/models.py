import logging
from typing import Text, List
from numpy.typing import NDArray

from .website import WebsiteParam
from .dataset import Dataset

logger = logging.getLogger(__name__)


class Model:
    def name(self) -> Text:
        raise NotImplementedError("You must implement the `name` method.")

    def fit(self) -> None:
        raise NotImplementedError("You must implement the `fit` method.")

    def predict(self, X, **kwargs) -> NDArray:
        raise NotImplementedError("You must implement the `predict` method.")

    def save_model(self) -> None:
        """Save a trained model into the file system before the server shuts down"""
        pass

    def load_model(self) -> None:
        """Load a trained model from the file system after the server starts up"""
        pass

    def website_params(self) -> List[WebsiteParam]:
        """Define custom parameters used during the prediction process"""
        return []

    def update_data(self, ds: Dataset) -> None:
        self.dataset = ds

    def to_dict(self):
        return {
            "name": self.name(),
            "params": [p.to_dict() for p in self.website_params()],
        }

    def __str__(self) -> Text:
        return f"Model '{self.name()}'"
