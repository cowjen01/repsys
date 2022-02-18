import configparser
import os

import repsys.constants as const
from repsys.utils import get_default_config_path


class DatasetConfig:
    def __init__(self, test_holdout_prop: float, train_split_prop: float,
                 user_interactions_threshold: int,
                 item_interactions_threshold: int, interaction_value_threshold: float):
        self.test_holdout_prop = test_holdout_prop
        self.train_split_prop = train_split_prop
        self.user_interactions_threshold = user_interactions_threshold
        self.item_interactions_threshold = item_interactions_threshold
        self.interaction_value_threshold = interaction_value_threshold


class Config:
    def __init__(self, seed: int, server_port: int, dataset_config: DatasetConfig):
        self.dataset = dataset_config
        self.seed = seed
        self.server_port = server_port


def read_config(config_path: str = None):
    config = configparser.ConfigParser()

    if not config_path:
        config_path = get_default_config_path()

    if not os.path.isfile(config_path):
        dataset_config = DatasetConfig(
            const.DEFAULT_TEST_HOLDOUT_PROP,
            const.DEFAULT_TRAIN_SPLIT_PROP,
            const.DEFAULT_USER_INTERACTIONS_THRESHOLD,
            const.DEFAULT_ITEM_INTERACTIONS_THRESHOLD,
            const.DEFAULT_INTERACTION_VALUE_THRESHOLD
        )

        return Config(const.DEFAULT_SEED, const.DEFAULT_SERVER_PORT, dataset_config)

    with open(config_path, 'r') as f:
        config.read_file(f)

        dataset_config = DatasetConfig(
            config.getfloat('dataset', 'TEST_HOLDOUT_PROP',
                            fallback=const.DEFAULT_TEST_HOLDOUT_PROP),
            config.getfloat('dataset', 'TRAIN_SPLIT_PROP',
                            fallback=const.DEFAULT_TRAIN_SPLIT_PROP),
            config.getint('dataset', 'USER_INTERACTIONS_THRESHOLD',
                          fallback=const.DEFAULT_USER_INTERACTIONS_THRESHOLD),
            config.getint('dataset', 'ITEM_INTERACTIONS_THRESHOLD',
                          fallback=const.DEFAULT_ITEM_INTERACTIONS_THRESHOLD),
            config.getfloat('dataset', 'INTERACTION_VALUE_THRESHOLD',
                            fallback=const.DEFAULT_INTERACTION_VALUE_THRESHOLD)
        )

        seed = config.getint('general', 'SEED', fallback=const.DEFAULT_SEED)
        server_port = config.get('server', 'PORT', fallback=const.DEFAULT_SERVER_PORT)

        return Config(seed, server_port, dataset_config)
