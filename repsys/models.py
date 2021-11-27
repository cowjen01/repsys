import logging
import os
import pickle
from typing import Text, List
from abc import ABC, abstractmethod

from repsys.website import PredictParam
from repsys.dataset import Dataset

logger = logging.getLogger(__name__)


class Model(ABC):
    @abstractmethod
    def name(self) -> Text:
        pass

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abstractmethod
    def save(self, dir_path) -> None:
        pass

    @abstractmethod
    def load(self, dir_path) -> None:
        pass

    def predict_params(self) -> List[PredictParam]:
        return []

    def update_data(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def to_dict(self):
        return {
            "name": self.name(),
            "params": [p.to_dict() for p in self.predict_params()],
        }

    def __str__(self) -> Text:
        return f"Model '{self.name()}'"


class ScikitModel(Model):
    def _checkpoint_path(self, dir_path) -> Text:
        return os.path.join(dir_path, self.name())

    def save(self, dir_path) -> None:
        checkpoint = open(self._checkpoint_path(dir_path), "wb")
        pickle.dump(self.model, checkpoint)

    def load(self, dir_path) -> None:
        self.model = pickle.load(open(self._checkpoint_path(dir_path), "rb"))
