import glob
import os
import shutil
import time
import numpy as np
import random
from typing import Text, Optional


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
    return latest_checkpoint("split-*.zip")


def latest_eval_checkpoint() -> Optional[Text]:
    return latest_checkpoint("eval-*.zip")


def new_split_checkpoint():
    create_checkpoints_dir()
    return os.path.join(
        checkpoints_dir_path(), fill_timestamp("split-{ts}.zip")
    )


def new_eval_checkpoint():
    create_checkpoints_dir()
    return os.path.join(checkpoints_dir_path(), fill_timestamp("eval-{ts}.zip"))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
