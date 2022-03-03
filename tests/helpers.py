import os
import pandas as pd


def get_fixtures_path():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(abs_path, "fixtures")


def load_items(file_name):
    path = os.path.join(get_fixtures_path(), "items", file_name)
    return pd.read_csv(path, header=0)


def load_interacts(file_name):
    path = os.path.join(get_fixtures_path(), "interacts", file_name)
    return pd.read_csv(path, header=0)
