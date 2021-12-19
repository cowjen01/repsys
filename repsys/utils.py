import os
import shutil
from typing import Text


def remove_dir(path: Text):
    shutil.rmtree(path)


def create_dir(path: Text):
    if not os.path.exists(path):
        os.makedirs(path)


def tmp_dir_path():
    return os.path.join(os.getcwd(), "tmp")


def create_tmp_dir():
    create_dir(tmp_dir_path())


def remove_tmp_dir():
    remove_dir(tmp_dir_path())


def unzip_dir(zip_path: Text, dir_path: Text):
    shutil.unpack_archive(zip_path, dir_path)


def zip_dir(zip_path: Text, dir_path: Text):
    path_split = zip_path.split(".")
    if path_split[-1] == "zip":
        zip_path = ".".join(path_split[:-1])

    shutil.make_archive(zip_path, "zip", dir_path)
