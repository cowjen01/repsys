import configparser
import os
from typing import List

import repsys.constants as const
from repsys.errors import InvalidConfigError


class DatasetConfig:
    def __init__(
        self,
        test_holdout_prop: float,
        train_split_prop: float,
        min_user_interacts: int,
        min_item_interacts: int,
    ):
        self.test_holdout_prop = test_holdout_prop
        self.train_split_prop = train_split_prop
        self.min_user_interacts = min_user_interacts
        self.min_item_interacts = min_item_interacts


class EvaluationConfig:
    def __init__(
        self,
        precision_recall_k: List[int],
        ndcg_k: List[int],
        coverage_k: List[int],
        diversity_k: List[int],
        novelty_k: List[int],
        percentage_lt_k: List[int],
        coverage_lt_k: List[int],
    ):
        self.precision_recall_k = precision_recall_k
        self.ndcg_k = ndcg_k
        self.coverage_k = coverage_k
        self.diversity_k = diversity_k
        self.novelty_k = novelty_k
        self.percentage_lt_k = percentage_lt_k
        self.coverage_lt_k = coverage_lt_k


class Config:
    def __init__(
        self,
        checkpoints_dir: str,
        seed: int,
        debug: bool,
        server_port: int,
        dataset_config: DatasetConfig,
        eval_config: EvaluationConfig,
    ):
        self.dataset = dataset_config
        self.eval = eval_config
        self.checkpoints_dir = checkpoints_dir
        self.debug = debug
        self.seed = seed
        self.server_port = server_port


def validate_dataset_config(config: DatasetConfig):
    if config.train_split_prop <= 0 or config.train_split_prop >= 1:
        raise InvalidConfigError("The train split proportion must be between 0 and 1")

    if config.test_holdout_prop <= 0 or config.test_holdout_prop >= 1:
        raise InvalidConfigError("The test holdout proportion must be between 0 and 1")

    if config.min_user_interacts < 0:
        raise InvalidConfigError("Minimum user interactions can be negative")

    if config.min_item_interacts < 0:
        raise InvalidConfigError("Minimum item interactions can be negative")


def parse_list(arg: str, sep: str = ","):
    if type(arg) == str:
        return [int(x.strip()) for x in arg.split(sep)]
    return arg


def read_config(config_path: str = None):
    config = configparser.ConfigParser()

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config.read_file(f)

    dataset_config = DatasetConfig(
        config.getfloat("dataset", "test_holdout_prop", fallback=const.DEFAULT_TEST_HOLDOUT_PROP),
        config.getfloat("dataset", "train_split_prop", fallback=const.DEFAULT_TRAIN_SPLIT_PROP),
        config.getint("dataset", "min_user_interacts", fallback=const.DEFAULT_MIN_USER_INTERACTS),
        config.getint("dataset", "min_item_interacts", fallback=const.DEFAULT_MIN_ITEM_INTERACTS),
    )

    validate_dataset_config(dataset_config)

    evaluator_config = EvaluationConfig(
        parse_list(
            config.get(
                "evaluation",
                "precision_recall_k",
                fallback=const.DEFAULT_PRECISION_RECALL_K,
            )
        ),
        parse_list(config.get("evaluation", "ndcg_k", fallback=const.DEFAULT_NDCG_K)),
        parse_list(config.get("evaluation", "coverage_k", fallback=const.DEFAULT_COVERAGE_K)),
        parse_list(config.get("evaluation", "diversity_k", fallback=const.DEFAULT_DIVERSITY_K)),
        parse_list(config.get("evaluation", "novelty_k", fallback=const.DEFAULT_NOVELTY_K)),
        parse_list(config.get("evaluation", "percentage_lt_k", fallback=const.DEFAULT_PERCENTAGE_LT_K)),
        parse_list(config.get("evaluation", "coverage_lt_k", fallback=const.DEFAULT_COVERAGE_LT_K)),
    )

    return Config(
        config.get("general", "checkpoints_dir", fallback=const.DEFAULT_CHECKPOINTS_DIR),
        config.getint("general", "seed", fallback=const.DEFAULT_SEED),
        config.getboolean("general", "debug", fallback=False),
        config.get("server", "port", fallback=const.DEFAULT_SERVER_PORT),
        dataset_config,
        evaluator_config,
    )
