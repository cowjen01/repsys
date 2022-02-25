import functools
import glob
import os
import random
import shutil
import time
from typing import Optional, List

import numpy as np


def remove_dir(path: str) -> None:
    shutil.rmtree(path)


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def current_dir_path() -> str:
    return os.getcwd()


def default_config_path() -> str:
    return os.path.join(current_dir_path(), 'repsys.ini')


def tmp_dir_path() -> str:
    return os.path.join(current_dir_path(), "tmp")


def create_tmp_dir() -> None:
    create_dir(tmp_dir_path())


def remove_tmp_dir() -> None:
    remove_dir(tmp_dir_path())


def unzip_dir(zip_path: str, dir_path: str) -> None:
    shutil.unpack_archive(zip_path, dir_path)


def zip_dir(zip_path: str, dir_path: str) -> None:
    path_chunks = zip_path.split(".")
    if path_chunks[-1] == "zip":
        zip_path = ".".join(path_chunks[:-1])

    shutil.make_archive(zip_path, "zip", dir_path)


def get_subclasses(cls) -> List[str]:
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in get_subclasses(s)
    ]


def fill_timestamp(file_name: str) -> str:
    if "{ts}" in file_name:
        ts = int(time.time())
        return file_name.format(ts=ts)

    return file_name


def checkpoints_dir_path() -> str:
    return ".repsys_checkpoints/"


def create_checkpoints_dir() -> None:
    create_dir(checkpoints_dir_path())


def latest_checkpoints(pattern: str, history: int = 0) -> Optional[str]:
    path = os.path.join(checkpoints_dir_path(), pattern)
    files = glob.glob(path)

    if not files or len(files) <= history:
        return None

    files.sort(reverse=True)

    return files[history]


def latest_split_checkpoint() -> Optional[str]:
    return latest_checkpoints("dataset-split-*.zip")


def latest_dataset_eval_checkpoint() -> Optional[str]:
    return latest_checkpoints("dataset-eval-*.zip")


def models_eval_checkpoints(history: int = 0) -> Optional[str]:
    return latest_checkpoints("models-eval-*.zip", history)


def new_split_checkpoint() -> str:
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("dataset-split-{ts}.zip"))


def new_dataset_eval_checkpoint() -> str:
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("dataset-eval-{ts}.zip"))


def new_models_eval_checkpoint() -> str:
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
