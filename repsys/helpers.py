import functools
import glob
import os
import random
import shutil
import time
from typing import List

import numpy as np

from repsys.constants import CURRENT_VERSION


def remove_dir(path: str) -> None:
    shutil.rmtree(path)


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def current_dir_path() -> str:
    return os.getcwd()


def default_config_path() -> str:
    return os.path.join(current_dir_path(), "repsys.ini")


def tmp_dir_path() -> str:
    return os.path.join(current_dir_path(), "tmp")


def checkpoints_dir_path() -> str:
    return os.path.join(current_dir_path(), ".repsys_checkpoints")


def create_checkpoints_dir() -> None:
    create_dir(checkpoints_dir_path())


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
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_subclasses(s)]


def current_ts() -> int:
    return int(time.time())


def find_checkpoints(dir_path: str, pattern: str, history: int = 1) -> List[str]:
    path = os.path.join(dir_path, pattern)
    files = glob.glob(path)

    if not files:
        return []

    files.sort(reverse=True)

    return files[:history]


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


def write_version(version: str, dir_name: str):
    with open(os.path.join(dir_name, "version.txt"), "w") as f:
        f.write(version)


def read_version(dir_name: str):
    path = os.path.join(dir_name, "version.txt")
    if not os.path.isfile(path):
        return CURRENT_VERSION

    with open(path, "r") as f:
        return f.readline()
