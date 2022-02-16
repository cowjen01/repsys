import functools
import logging
import os
from typing import List, Text

import click

from repsys.constants import DEFAULT_SERVER_PORT
from repsys.dataset import Dataset
from repsys.evaluators import ModelEvaluator
from repsys.loaders import load_dataset_pkg, load_models_pkg
from repsys.model import Model
from repsys.server import run_server
from repsys.utils import (
    create_dir,
    create_tmp_dir,
    latest_split_checkpoint,
    new_split_checkpoint,
    latest_eval_checkpoint,
    new_eval_checkpoint,
    remove_tmp_dir,
    tmp_dir_path,
    zip_dir,
)

logger = logging.getLogger(__name__)


def split_input_callback(ctx, param, value):
    if not value:
        path = latest_split_checkpoint()

        if not path:
            raise click.ClickException(
                "No split were found in the default directory '.repsys_checkpoints'. "
                "Please provide a path to the split or run 'repsys split' command."
            )

        return path

    return value


def eval_input_callback(ctx, param, value):
    if not value:
        path = latest_eval_checkpoint()

        if not path:
            raise click.ClickException(
                "No evaluation found in the default directory '.repsys_checkpoints'. "
                "Please provide a path to the evaluation or run 'repsys eval' command."
            )

        return path

    return value


def split_output_callback(ctx, param, value):
    if not value:
        return new_split_checkpoint()
    return value


def eval_output_callback(ctx, param, value):
    if not value:
        return new_eval_checkpoint()
    return value


def models_callback(ctx, param, value):
    return load_models_pkg(value)


def dataset_callback(ctx, param, value):
    return load_dataset_pkg(value)


def dataset_option(func):
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


def models_option(func):
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


def split_option(func):
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


def fit_models(models: List[Model], dataset: Dataset, training: bool):
    for model in models:
        if training:
            logger.info(f"Training model '{model.name()}' ...")
        else:
            logger.info(f"Fitting model '{model.name()}' ...")

        model.update_dataset(dataset)
        model.fit(training=training)


def evaluate_models(
    models: List[Model], dataset: Dataset, data_type: Text, output_path: Text
):
    create_tmp_dir()

    evaluator = ModelEvaluator()
    evaluator.update_dataset(dataset)
    for model in models:
        logger.info(f"Evaluating model '{model.name()}' ...")

        if data_type == "test":
            evaluator.test_model_eval(model)
        else:
            evaluator.vad_model_eval(model)

        evaluator.print()

        eval_dir = os.path.join(tmp_dir_path(), model.name())
        create_dir(eval_dir)
        evaluator.save(eval_dir)

    zip_dir(output_path, tmp_dir_path())
    remove_tmp_dir()


@click.group()
def repsys():
    """Repsys client for recommendation systems development."""
    pass


@repsys.command()
@models_option
@dataset_option
@split_option
@click.option(
    "-t",
    "--data-type",
    default="test",
    type=click.Choice(["test", "vad"]),
    show_default=True,
)
@click.option("-o", "--output-path", callback=eval_output_callback)
def evaluate(
    models: List[Model],
    dataset: Dataset,
    split_path: Text,
    data_type: Text,
    output_path: Text,
):
    """Evaluate implemented models."""
    dataset.load(split_path)
    fit_models(models, dataset, training=False)
    evaluate_models(models, dataset, data_type, output_path)


@repsys.command()
@models_option
@dataset_option
@split_option
@click.option(
    "-e",
    "--eval-path",
    callback=eval_input_callback,
    type=click.Path(exists=True),
)
@click.option(
    "-p", "--port", default=DEFAULT_SERVER_PORT, type=int, show_default=True
)
def server(
    models: List[Model],
    dataset: Dataset,
    split_path: Text,
    eval_path: Text,
    port: int,
):
    """Start Repsys server."""
    # create_tmp_dir()

    # unzip_dir(eval_path, tmp_dir_path())

    # evaluator = ModelEvaluator()
    # evaluator.load(tmp_dir_path())

    # remove_tmp_dir()

    # evaluator.print()
    dataset.load(split_path)
    fit_models(models, dataset, training=False)
    run_server(port, models, dataset)


@repsys.command()
@dataset_option
@models_option
@split_option
def train(models: List[Model], dataset: Dataset, split_path: Text):
    """Train models by providing dataset split."""
    dataset.load(split_path)
    fit_models(models, dataset, training=True)


@repsys.command()
@dataset_option
@click.option("-o", "--output-path", callback=split_output_callback)
def split(dataset: Dataset, output_path: Text):
    """Create a train/validation/test split."""
    logger.info("Creating splits of the input data ...")
    dataset.fit()

    logger.info(f"Saving the split into '{output_path}'")
    dataset.save(output_path)
