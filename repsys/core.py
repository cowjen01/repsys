from typing import Dict, Text
import logging
import sys
import os
import numpy as np
from scipy import sparse
import time
import shutil
import glob

from repsys.models import Model
from repsys.dataset import Dataset
from repsys.evaluator import Evaluator
from repsys.utils import remove_dir, create_dir


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

    def _checkpoints_dir_path(self):
        return os.path.join(os.getcwd(), ".repsys", "checkpoints")

    def _tmp_dir_path(self):
        return os.path.join(os.getcwd(), ".repsys", "tmp")

    def _checkpoints_zip_files(self):
        dir_path = self._checkpoints_dir_path()
        return glob.glob(os.path.join(dir_path, "*.zip"))

    def load_models(self) -> None:
        zip_files = self._checkpoints_zip_files()

        if len(zip_files) == 0:
            logger.error("There are no checkpoints to unzip.")
            sys.exit(1)

        zip_files.sort(reverse=True)

        create_dir(self._tmp_dir_path())

        shutil.unpack_archive(zip_files[0], self._tmp_dir_path())

        for model in self.models.values():
            logger.info(f"Loading model called '{model.name()}'.")

            try:
                model.load(self._tmp_dir_path())
            except Exception:
                logger.error(
                    f"Model called '{model.name()}' has not been trained yet."
                )
                remove_dir(self._tmp_dir_path())
                sys.exit(1)

        remove_dir(self._tmp_dir_path())

    def save_models(self) -> None:
        create_dir(self._tmp_dir_path())

        for model in self.models.values():
            model.save(self._tmp_dir_path())

        create_dir(self._checkpoints_dir_path())

        zip_file_name = str(int(time.time()))
        zip_file_path = os.path.join(self._checkpoints_dir_path(), zip_file_name)

        shutil.make_archive(zip_file_path, "zip", self._tmp_dir_path())

        remove_dir(self._tmp_dir_path())

    def eval_models(self) -> None:
        evaluator = Evaluator()

        for model in self.models.values():
            evaluator.evaluate_model(
                model, self.dataset.vad_data_tr, self.dataset.vad_data_te
            )

        evaluator.print_results()

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
