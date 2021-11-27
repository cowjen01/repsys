from typing import Dict, Text
import logging
import sys
import numpy as np
from scipy import sparse

from repsys.models import Model
from repsys.dataset import Dataset


logger = logging.getLogger(__name__)


class RepsysCore:
    def __init__(self, models: Dict[Text, Model], dataset: Dataset) -> None:
        self.models = models
        self.dataset = dataset

    def update_models_dataset(self) -> None:
        for model in self.models.values():
            model.update_data(self.dataset)

    def train_models(self) -> None:
        for model in self.models.values():
            logger.info(f"Training model called '{model.name()}'.")
            model.fit()

    def load_models(self) -> None:
        for model in self.models.values():
            logger.info(f"Loading model called '{model.name()}'.")

            if model.model_trained():
                model.load_model()
            else:
                logger.error(
                    f"Model called '{model.name()}' has not been trained yet."
                )
                sys.exit(1)

    def eval_models(self) -> None:
        for model in self.models.values():
            pass

    def save_models(self) -> None:
        for model in self.models.values():
            model.save_model()

    def get_model(self, model_name):
        return self.models.get(model_name)

    def filter_items(self, column, query):
        filter = self.dataset.items[column].str.contains(query, case=False)
        return self.dataset.items[filter]

    def get_interacted_items(self, user_index):
        interactions = self.dataset.vad_data_tr[user_index]
        return self.dataset.items.loc[(interactions > 0).indices]

    def get_user_history(self, user_index):
        return self.dataset.vad_data_tr[user_index]

    def input_from_interactions(self, interactions):
        return sparse.csr_matrix(
            (
                np.ones_like(interactions),
                (np.zeros_like(interactions), interactions),
            ),
            dtype="float64",
            shape=(1, self.dataset.n_items),
        )

    def prediction_to_items(self, prediction, limit=20):
        idxs = (-prediction[0]).argsort()[:limit]
        return self.dataset.items.loc[idxs]
