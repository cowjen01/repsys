import os
import time
import glob
import logging
from typing import Text, Optional

from repsys.utils import create_dir


logger = logging.getLogger(__name__)


def fill_timestamp(str: Text):
    if "{ts}" in str:
        ts = int(time.time())
        return str.format(ts=ts)

    return str


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


def new_split_checkpoint():
    create_checkpoints_dir()

    path = fill_timestamp("split-{ts}.zip")
    path = os.path.join(checkpoints_dir_path(), path)

    return path
