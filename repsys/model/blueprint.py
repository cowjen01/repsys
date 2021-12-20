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
    def save(self, dir_path: Text) -> None:
        pass

    @abstractmethod
    def load(self, dir_path: Text) -> None:
        pass

    def predict_params(self) -> List[PredictParam]:
        return []

    def update_data(self, dataset: Dataset) -> None:
        if not isinstance(dataset, Dataset):
            raise Exception("Data must be an instance of the Dataset class.")

        self.dataset = dataset

    def to_dict(self):
        return {
            "name": self.name(),
            "params": [p.to_dict() for p in self.predict_params()],
        }

    def __str__(self) -> Text:
        return f"Model '{self.name()}'"


class ScikitModel(Model):
    def save(self, path: Text) -> None:
        checkpoint = open(path, "wb")
        pickle.dump(self.serialize(), checkpoint)

    def load(self, path: Text) -> None:
        checkpoint = pickle.load(open(path, "rb"))
        self.unserialize(checkpoint)

    def serialize(self):
        return self.model

    def unserialize(self, state) -> None:
        self.model = state
