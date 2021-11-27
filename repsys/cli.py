import click
import logging
import sys
import functools

from typing import List

from repsys.core import RepsysCore
from repsys.models import Model
from repsys.server import run_server
from repsys.loader import ClassLoader
from repsys.dataset import Dataset
from repsys.constants import DEFAULT_SERVER_PORT

logger = logging.getLogger(__name__)


def load_models(models_package) -> List[Model]:
    model_loader = ClassLoader(Model)
    model_loader.register_package(models_package)

    if len(model_loader.instances) == 0:
        logger.exception(
            "At least one instance of Model class must be defined."
        )
        sys.exit(1)

    return model_loader.instances


def load_dataset(dataset_package) -> Dataset:
    dataset_loader = ClassLoader(Dataset)
    dataset_loader.register_package(dataset_package)

    if len(dataset_loader.instances) != 1:
        logger.exception("One instance of Dataset class must be defined.")
        sys.exit(1)

    return list(dataset_loader.instances.values())[0]


def init_core(models_package, dataset_package):
    dataset = load_dataset(dataset_package)
    dataset.load_dataset()

    models = load_models(models_package)

    core = RepsysCore(models=models, dataset=dataset)

    core.update_models_dataset()

    return core


def common_params(func):
    @click.option(
        "-m", "--models", "models_package", default="models", show_default=True
    )
    @click.option(
        "-d",
        "--dataset",
        "dataset_package",
        default="dataset",
        show_default=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@click.group()
def repsys():
    """Repsys client for managing of recommender models."""
    pass


@repsys.command()
@common_params
def train(models_package, dataset_package):
    """Train implemented models."""
    core = init_core(models_package, dataset_package)
    core.train_models()
    core.save_models()


@repsys.command()
@common_params
def evaluate(models_package, dataset_package):
    """Evaluate trained models."""
    core = init_core(models_package, dataset_package)
    core.load_models()
    core.eval_models()


@repsys.command()
@common_params
@click.option(
    "-p", "--port", "port", default=DEFAULT_SERVER_PORT, show_default=True
)
def server(port, models_package, dataset_package):
    """Start Repsys server."""

    core = init_core(models_package, dataset_package)
    core.load_models()

    run_server(port=port, core=core)
