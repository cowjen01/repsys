import logging
from typing import Text, List
from abc import ABC, abstractmethod
import functools

from repsys.web import WebParam
from repsys.dataset import Dataset

logger = logging.getLogger(__name__)


def enforcedataset(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "dataset") or self.dataset is None:
            raise Exception("The model must be updated with a dataset.")
        return func(self, *args, **kwargs)

    return wrapper


class Model(ABC):
    @abstractmethod
    def name(self) -> Text:
        """Get a unique name of the model."""
        pass

    @abstractmethod
    @enforcedataset
    def fit(self, training: bool = False) -> None:
        """Train the model using the training interactions.
        If the model is already trained, load it from a file."""
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """Make a prediction from the input interactions and
        return a matrix including ratings for each item. The second
        argument includes a dictionary of values for each parameter
        set in the web application."""
        pass

    def web_params(self) -> List[WebParam]:
        """Return a list of parameters that will be displayed
        in the web application allowing a user to pass additional
        arguments to the prediction method."""
        return []

    def update_dataset(self, dataset: Dataset) -> None:
        """Update the model with an instance of the dataset."""
        if not isinstance(dataset, Dataset):
            raise Exception(
                "The data must be an instance of the dataset class."
            )

        self.dataset = dataset

    @enforcedataset
    def recommend_top_items(self, X, limit=20, **kwargs):
        """Make a prediction, but return directly a list of items."""
        prediction = self.predict(X, **kwargs)
        idxs = (-prediction[0]).argsort()[:limit]
        return idxs

    def to_dict(self):
        """Serialize details about the model to the dictionary."""
        return {
            "name": self.name(),
            "params": [p.to_dict() for p in self.web_params()],
        }

    def __str__(self) -> Text:
        return f"Model '{self.name()}'"
