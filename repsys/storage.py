import sys
import os
import time
import shutil
import glob
import logging
from typing import Dict, Text

from repsys.utils import remove_dir, create_dir
from repsys.model import Model
from repsys.dataset import Dataset


logger = logging.getLogger(__name__)


class CheckpointStorage:
    def __init__(
        self,
        models: Dict[Text, Model],
        dataset: Dataset,
        dirname: Text = ".repsys_checkpoints",
    ) -> None:
        self.models = models
        self.dataset = dataset
        self.dir_path = os.path.join(os.getcwd(), dirname)

    def _checkpoints_dir_path(self):
        return os.path.join(self.dir_path, "models")

    def _tmp_dir_path(self):
        return os.path.join(self.dir_path, "tmp")

    def _checkpoints_zip_files(self):
        dir_path = self._checkpoints_dir_path()
        return glob.glob(os.path.join(dir_path, "*.zip"))

    def load_all(self) -> None:
        zip_files = self._checkpoints_zip_files()

        if len(zip_files) == 0:
            logger.error("There are no model's checkpoints to unzip.")
            sys.exit(1)

        zip_files.sort(reverse=True)

        create_dir(self._tmp_dir_path())

        shutil.unpack_archive(zip_files[0], self._tmp_dir_path())

        for model in self.models.values():
            logger.info(f"Loading model '{model.name()}'.")

            try:
                model.load(self._tmp_dir_path())
            except Exception:
                logger.error(
                    f"Model '{model.name()}' has not been trained yet."
                )
                remove_dir(self._tmp_dir_path())
                sys.exit(1)

        remove_dir(self._tmp_dir_path())

    def save_all(self) -> None:
        create_dir(self._tmp_dir_path())

        for model in self.models.values():
            model.save(self._tmp_dir_path())

        create_dir(self._checkpoints_dir_path())

        zip_file_name = str(int(time.time()))
        zip_file_path = os.path.join(
            self._checkpoints_dir_path(), zip_file_name
        )

        shutil.make_archive(zip_file_path, "zip", self._tmp_dir_path())

        remove_dir(self._tmp_dir_path())

    def _get_dataset_zip_path(self):
        return os.path.join(self.dir_path, f'{self.dataset.name()}.zip')

    def load_dataset(self):
        zip_path = self._get_dataset_zip_path()

        if os.path.exists(zip_path):
            self.dataset.load(zip_path)
        else:
            self.dataset.fit()
            self.dataset.save(zip_path)
