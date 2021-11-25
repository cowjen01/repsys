import click
import logging
import sys

from typing import List

from repsys.core import RepsysCore

from repsys.models import Model
from repsys.server import run_server
from repsys.loader import ClassLoader
from repsys.dataset import Dataset
from repsys.constants import DEFAULT_SERVER_PORT

logger = logging.getLogger(__name__)


@click.group()
def repsys():
    """Repsys client for managing of recommender models."""
    pass


@repsys.command()
@click.option(
    "-p", "--port", "port", default=DEFAULT_SERVER_PORT, show_default=True
)
@click.option(
    "-m", "--models", "models_package", default="models", show_default=True
)
@click.option(
    "-d", "--dataset", "dataset_package", default="dataset", show_default=True
)
def server(port, models_package, dataset_package):
    """Start Repsys server."""

    dataset_loader = ClassLoader(Dataset)
    dataset_loader.register_package(dataset_package)

    if len(dataset_loader.instances) != 1:
        logger.exception("One instance of Dataset class must be defined.")
        sys.exit(1)

    dataset: Dataset = list(dataset_loader.instances.values())[0]
    dataset.load_dataset()

    model_loader = ClassLoader(Model)
    model_loader.register_package(models_package)

    if len(model_loader.instances) == 0:
        logger.exception(
            "At least one instance of Model class must be defined."
        )
        sys.exit(1)

    models: List[Model] = model_loader.instances

    core = RepsysCore(models=models, dataset=dataset)

    core.init_models()

    run_server(port=port, core=core)
