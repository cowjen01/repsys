import os
import shutil


def remove_dir(path):
    shutil.rmtree(path)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
