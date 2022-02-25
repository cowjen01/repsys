import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np
from scipy.sparse import csr_matrix

from repsys.dataset import Dataset
from repsys.helpers import enforce_updated
from repsys.web import WebParam

logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self):
        self.dataset: Optional[Dataset] = None
        self._updated = False

    @abstractmethod
    def name(self) -> str:
        """Get a unique name of the model."""
        pass

    @abstractmethod
    @enforce_updated
    def fit(self, training: bool = False) -> None:
        """Train the model using the training interactions.
        If the model is already trained, load it from a file."""
        pass

    @abstractmethod
    @enforce_updated
    def predict(self, x: csr_matrix, **kwargs):
        """Make a prediction from the input interactions and
        return a matrix including ratings for each item. The second
        argument includes a dictionary of values for each parameter
        set in the web application."""
        pass

    def web_params(self) -> Dict[str, Type[WebParam]]:
        """Return a list of parameters that will be displayed
        in the web application allowing a user to pass additional
        arguments to the prediction method."""
        return {}

    def update_dataset(self, dataset: Dataset) -> None:
        """Update the model with an instance of the dataset."""
        self._updated = True
        self.dataset = dataset

    @enforce_updated
    def predict_top_items(self, x: csr_matrix, n=20, **kwargs):
        """Make a prediction, but return directly a list of top ids."""
        prediction = self.predict(x, **kwargs)
        indices = (-prediction).argsort()[:, :n]
        return np.vectorize(self.dataset.item_index_to_id)(indices)

    def to_dict(self):
        """Serialize details about the model to the dictionary."""
        return {"params": {
            key: param.to_dict() for key, param in self.web_params().items()
        }}

    def __str__(self) -> str:
        return f"Model '{self.name()}'"
