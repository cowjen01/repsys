from typing import Dict, Text
import click
import logging
import functools
from repsys.dataset import Dataset

from repsys.server import run_server
from repsys.model import Model
from repsys.loaders import load_dataset_pkg, load_models_pkg
from repsys.constants import DEFAULT_SERVER_PORT
from repsys.checkpoints import (
    latest_split_checkpoint,
    new_split_checkpoint,
)

logger = logging.getLogger(__name__)


def split_input_callback(ctx, param, value):
    if not value:
        path = latest_split_checkpoint()

        if not path:
            raise click.ClickException(
                "No split was found in the default directory '.repsys_checkpoints'. "
                "Please provide a path to the split or run 'repsys split' command."
            )

        return path

    return value


def split_output_callback(ctx, param, value):
    if not value:
        return new_split_checkpoint()

    return value


def models_callback(ctx, param, value):
    return load_models_pkg(value)


def dataset_callback(ctx, param, value):
    return load_dataset_pkg(value)


def datasetoption(func):
    @click.option(
        "-d",
        "--dataset-pkg",
        "dataset",
        callback=dataset_callback,
        default="dataset",
        show_default=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def modelsoption(func):
    @click.option(
        "-m",
        "--models-pkg",
        "models",
        callback=models_callback,
        default="models",
        show_default=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def splitoption(func):
    @click.option(
        "-s",
        "--split-path",
        callback=split_input_callback,
        type=click.Path(exists=True),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def fit_models(models: Dict[Text, Model], dataset: Dataset):
    logger.info("Getting models ready ...")
    for model in models.values():
        logger.info(f"Fitting model '{model.name()}' ...")

        model.update_dataset(dataset)
        model.fit(training=True)


@click.group()
def repsys():
    """Repsys client for recommendation systems development."""
    pass


# @repsys.command()
# @packageoptions
# def evaluate(models_package, dataset_package):
#     """Evaluate trained models."""
#     core = init_core(models_package, dataset_package)
#     core.load_models_checkpoint()
#     core.eval_models()


@repsys.command()
@modelsoption
@datasetoption
@splitoption
@click.option(
    "-p", "--port", default=DEFAULT_SERVER_PORT, type=int, show_default=True
)
def server(
    models: Dict[Text, Model], dataset: Dataset, split_path: Text, port: int
):
    """Start Repsys server."""
    dataset.load(split_path)
    fit_models(models, dataset)

    run_server(port, models, dataset)


@repsys.command()
@datasetoption
@modelsoption
@splitoption
def train(models: Dict[Text, Model], dataset: Dataset, split_path: Text):
    """Train models by providing dataset split."""
    dataset.load(split_path)

    fit_models(models, dataset)


@repsys.command()
@datasetoption
@click.option("-o", "--output-path", callback=split_output_callback)
def split(dataset: Dataset, output_path: Text):
    """Create a train/validation/test split."""
    logger.info("Creating splits of the input data ...")
    dataset.fit()

    logger.info(f"Saving the split into '{output_path}'")
    dataset.save(output_path)
