import functools
import glob
import os
import random
import shutil
import time
from typing import Text, Optional

import numpy as np


def remove_dir(path: Text):
    shutil.rmtree(path)


def create_dir(path: Text):
    if not os.path.exists(path):
        os.makedirs(path)


def get_current_dir():
    return os.getcwd()


def get_default_config_path():
    return os.path.join(get_current_dir(), 'repsys.ini')


def tmp_dir_path():
    return os.path.join(get_current_dir(), "tmp")


def create_tmp_dir():
    create_dir(tmp_dir_path())


def remove_tmp_dir():
    remove_dir(tmp_dir_path())


def unzip_dir(zip_path: Text, dir_path: Text):
    shutil.unpack_archive(zip_path, dir_path)


def zip_dir(zip_path: Text, dir_path: Text):
    path_chunks = zip_path.split(".")
    if path_chunks[-1] == "zip":
        zip_path = ".".join(path_chunks[:-1])

    shutil.make_archive(zip_path, "zip", dir_path)


def get_subclasses(cls):
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in get_subclasses(s)
    ]


def fill_timestamp(file_name: Text):
    if "{ts}" in file_name:
        ts = int(time.time())
        return file_name.format(ts=ts)

    return file_name


def checkpoints_dir_path():
    return ".repsys_checkpoints/"


def create_checkpoints_dir():
    create_dir(checkpoints_dir_path())


def latest_checkpoint(pattern: Text) -> Optional[Text]:
    path = os.path.join(checkpoints_dir_path(), pattern)
    files = glob.glob(path)

    if not files:
        return None

    files.sort(reverse=True)

    return files[0]


def latest_split_checkpoint() -> Optional[Text]:
    return latest_checkpoint("dataset-split-*.zip")


def latest_dataset_eval_checkpoint() -> Optional[Text]:
    return latest_checkpoint("dataset-eval-*.zip")


def latest_models_eval_checkpoint() -> Optional[Text]:
    return latest_checkpoint("models-eval-*.zip")


def new_split_checkpoint():
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("dataset-split-{ts}.zip"))


def new_dataset_eval_checkpoint():
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("dataset-eval-{ts}.zip"))


def new_models_eval_checkpoint():
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("models-eval-{ts}.zip"))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def enforce_updated(func):
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if not getattr(self, "_updated"):
            raise Exception("The instance must be updated (call appropriate update method).")
        return func(self, *args, **kwargs)

    return _wrapper


def tmpdir_provider(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        create_tmp_dir()
        try:
            func(*args, **kwargs)
        finally:
            remove_tmp_dir()

    return _wrapper
