import functools
import logging
import os.path
from typing import Dict

import click
import numpy as np

from repsys.constants import DEFAULT_SERVER_PORT
from repsys.dataset import Dataset
from repsys.evaluators import DatasetEvaluator
from repsys.helpers import (
    latest_split_checkpoint,
    latest_dataset_eval_checkpoint,
    new_split_checkpoint,
    new_dataset_eval_checkpoint, create_tmp_dir, remove_tmp_dir, tmp_dir_path, zip_dir, unzip_dir,
)
from repsys.loaders import load_dataset_pkg, load_models_pkg
from repsys.model import Model
from repsys.server import run_server

logger = logging.getLogger(__name__)


def split_input_callback(ctx, param, value):
    if not value:
        path = latest_split_checkpoint()
        if not path:
            raise click.ClickException(
                "No split were found in the default directory '.repsys_checkpoints'. "
                "Please provide a path to the split or run 'repsys dataset split' command."
            )
        return path
    return value


def dataset_eval_input_callback(ctx, param, value):
    if not value:
        return latest_dataset_eval_checkpoint()
    return value


def split_output_callback(ctx, param, value):
    if not value:
        return new_split_checkpoint()
    return value


def dataset_eval_output_callback(ctx, param, value):
    if not value:
        return new_dataset_eval_checkpoint()
    return value


def models_callback(ctx, param, value):
    return load_models_pkg(value)


def dataset_callback(ctx, param, value):
    return load_dataset_pkg(value)


def dataset_pkg_option(func):
    @click.option("-d", "--dataset-pkg", "dataset", callback=dataset_callback, default="dataset", show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def models_pkg_option(func):
    @click.option("-m", "--models-pkg", "models", callback=models_callback, default="models", show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def split_path_option(func):
    @click.option("-s", "--split-path", callback=split_input_callback, type=click.Path(exists=True))
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def fit_models(models: Dict[str, Model], dataset: Dataset, training: bool):
    for model in models.values():
        if training:
            logger.info(f"Training model '{model.name()}' ...")
        else:
            logger.info(f"Fitting model '{model.name()}' ...")

        model.update_dataset(dataset)
        model.fit(training=training)


# def evaluate_models(models: Dict[str, Model], dataset: Dataset, data_type: str, output_path: str):
#     create_tmp_dir()
#
#     evaluator = ModelEvaluator()
#     evaluator.update_dataset(dataset)
#     for model in models.values():
#         logger.info(f"Evaluating model '{model.name()}' ...")
#
#         if data_type == "test":
#             evaluator.test_model_eval(model)
#         else:
#             evaluator.vad_model_eval(model)
#
#         evaluator.print()
#
#         eval_dir = os.path.join(tmp_dir_path(), model.name())
#         create_dir(eval_dir)
#         evaluator.save(eval_dir)
#
#     zip_dir(output_path, tmp_dir_path())
#     remove_tmp_dir()


def evaluate_dataset(dataset: Dataset, output_path: str):
    create_tmp_dir()
    try:
        evaluator = DatasetEvaluator()
        evaluator.update(dataset)

        item_embeddings, item_indexes = evaluator.get_item_embeddings()
        np.save(os.path.join(tmp_dir_path(), 'item-embeddings'), item_embeddings)
        np.save(os.path.join(tmp_dir_path(), 'item-indexes'), item_indexes)

        user_embeddings, user_indexes = evaluator.get_user_embeddings()
        np.save(os.path.join(tmp_dir_path(), 'user-embeddings'), user_embeddings)
        np.save(os.path.join(tmp_dir_path(), 'user-indexes'), user_indexes)

        zip_dir(output_path, tmp_dir_path())
    finally:
        remove_tmp_dir()


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def repsys_group(ctx, debug):
    """Repsys client for recommendation systems development."""
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@repsys_group.command(name='server')
@models_pkg_option
@dataset_pkg_option
@split_path_option
@click.option("--dataset-eval-path", callback=dataset_eval_input_callback, type=click.Path(exists=True))
@click.option("-p", "--port", default=DEFAULT_SERVER_PORT, type=int, show_default=True)
def server_start_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str, dataset_eval_path: str, port: int):
    """Start server."""
    dataset.load(split_path)
    fit_models(models, dataset, training=False)

    create_tmp_dir()
    try:
        unzip_dir(dataset_eval_path, tmp_dir_path())

        item_embeddings = np.load(os.path.join(tmp_dir_path(), 'item-embeddings.npy'))
        item_indexes = np.load(os.path.join(tmp_dir_path(), 'item-indexes.npy'))

        user_embeddings = np.load(os.path.join(tmp_dir_path(), 'user-embeddings.npy'))
        user_indexes = np.load(os.path.join(tmp_dir_path(), 'user-indexes.npy'))

        embeddings = {
            'train': {
                'items': (item_embeddings, item_indexes),
                'users': (user_embeddings, user_indexes)
            }
        }

    finally:
        remove_tmp_dir()

    run_server(port, models, dataset, embeddings)


@click.group(name='models')
def models_group():
    """Model commands."""
    """Implemented models commands."""
    pass


@click.group(name='dataset')
def dataset_group():
    """Dataset commands."""
    pass


repsys_group.add_command(dataset_group)
repsys_group.add_command(models_group)


# MODELS GROUP
# @models_group.command(name='eval')
# @models_pkg_option
# @dataset_pkg_option
# @split_path_option
# @click.option("-t", "--data-type", default="test", type=click.Choice(["test", "validation"]), show_default=True)
# @click.option("-o", "--output-path", callback=dataset_eval_output_callback)
# def models_eval_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str, data_type: str, output_path: str):
#     """Evaluate models using test/validation split."""
#     dataset.load(split_path)
#     fit_models(models, dataset, training=False)
#     # evaluate_models(models, dataset, data_type, output_path)


@models_group.command(name='train')
@dataset_pkg_option
@models_pkg_option
@split_path_option
def models_train_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str):
    """Train models using train split."""
    dataset.load(split_path)
    fit_models(models, dataset, training=True)


# DATASET GROUP
@dataset_group.command(name='split')
@dataset_pkg_option
@click.option("-o", "--output-path", callback=split_output_callback)
def dataset_split_cmd(dataset: Dataset, output_path: str):
    """Create train/validation/test split."""
    logger.info("Creating splits of the input data ...")
    dataset.prepare()

    logger.info(f"Saving the split into '{output_path}'")
    dataset.save(output_path)


@dataset_group.command(name='eval')
@dataset_pkg_option
@split_path_option
@click.option("-o", "--output-path", callback=dataset_eval_output_callback)
def dataset_eval_cmd(dataset: Dataset, split_path: str, output_path: str):
    """Create dataset embeddings using train split."""
    dataset.load(split_path)
    evaluate_dataset(dataset, output_path)
