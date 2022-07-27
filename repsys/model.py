import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

from repsys.config import Config
from repsys.dataset import Dataset
from repsys.ui import WebParam

logger = logging.getLogger(__name__)


class Model(ABC):
    dataset: Dataset = None
    config: Config = None

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(self, training: bool = False) -> None:
        pass

    @abstractmethod
    def predict(self, X: csr_matrix, **kwargs):
        pass

    def compute_embeddings(self, X: csr_matrix) -> Tuple[ndarray, ndarray]:
        raise Exception("You must implement your custom embeddings method.")

    def update(self, dataset: Dataset, config: Config):
        self.dataset = dataset
        self.config = config

    def web_params(self) -> Dict[str, WebParam]:
        return {}

    def predict_top_items(self, X: csr_matrix, n=20, **kwargs):
        prediction = self.predict(X, **kwargs)
        indices = (-prediction).argsort()[:, :n]
        return np.vectorize(self.dataset.item_index_to_id)(indices)

    def to_dict(self):
        return {"params": {key: param.to_dict() for key, param in self.web_params().items()}}

    def __str__(self) -> str:
        return f"Model '{self.name()}'"
