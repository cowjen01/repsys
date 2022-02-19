import configparser
import os

import repsys.constants as const
from repsys.helpers import get_default_config_path
from repsys.errors import InvalidConfigError


class DatasetConfig:
    def __init__(self, test_holdout_prop: float, train_split_prop: float, min_user_interacts: int,
                 min_item_interacts: int, min_interact_value: float):
        self.test_holdout_prop = test_holdout_prop
        self.train_split_prop = train_split_prop
        self.min_user_interacts = min_user_interacts
        self.min_item_interacts = min_item_interacts
        self.min_interact_value = min_interact_value

    def validate(self):
        if self.train_split_prop <= 0 or self.train_split_prop >= 1:
            raise InvalidConfigError('The train split proportion must be between 0 and 1')

        if self.test_holdout_prop <= 0 or self.test_holdout_prop >= 1:
            raise InvalidConfigError('The test holdout proportion must be between 0 and 1')

        if self.min_user_interacts < 0:
            raise InvalidConfigError('Minimum user interactions can be negative')

        if self.min_item_interacts < 0:
            raise InvalidConfigError('Minimum item interactions can be negative')


class Config:
    def __init__(self, seed: int, server_port: int, dataset_config: DatasetConfig):
        self.dataset = dataset_config
        self.seed = seed
        self.server_port = server_port


def read_config(config_path: str = None):
    config = configparser.ConfigParser()

    if not config_path:
        config_path = get_default_config_path()

    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config.read_file(f)

    dataset_config = DatasetConfig(
        config.getfloat('dataset', 'testHoldoutProp', fallback=const.DEFAULT_TEST_HOLDOUT_PROP),
        config.getfloat('dataset', 'trainSplitProp', fallback=const.DEFAULT_TRAIN_SPLIT_PROP),
        config.getint('dataset', 'minUserInteracts', fallback=const.DEFAULT_MIN_USER_INTERACTS),
        config.getint('dataset', 'minItemInteracts', fallback=const.DEFAULT_MIN_ITEM_INTERACTS),
        config.getfloat('dataset', 'minInteractValue', fallback=const.DEFAULT_MIN_INTERACT_VALUE)
    )

    dataset_config.validate()

    seed = config.getint('general', 'seed', fallback=const.DEFAULT_SEED)
    server_port = config.get('server', 'port', fallback=const.DEFAULT_SERVER_PORT)

    return Config(seed, server_port, dataset_config)
