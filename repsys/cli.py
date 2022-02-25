import functools
import logging
from typing import Dict

import click

from repsys.core import train_models, evaluate_dataset, start_server, split_dataset, evaluate_models
from repsys.dataset import Dataset
from repsys.helpers import (
    latest_split_checkpoint,
    latest_dataset_eval_checkpoint,
    new_split_checkpoint,
    new_dataset_eval_checkpoint,
    new_models_eval_checkpoint,
    models_eval_checkpoints
)
from repsys.loaders import load_packages
from repsys.model import Model

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


def latest_model_eval_input_callback(ctx, param, value):
    if not value:
        return models_eval_checkpoints(history=0)
    return value


def compare_model_eval_input_callback(ctx, param, value):
    if not value:
        return models_eval_checkpoints(history=1)
    return value


def split_output_callback(ctx, param, value):
    if not value:
        return new_split_checkpoint()
    return value


def dataset_eval_output_callback(ctx, param, value):
    if not value:
        return new_dataset_eval_checkpoint()
    return value


def models_eval_output_callback(ctx, param, value):
    if not value:
        return new_models_eval_checkpoint()
    return value


def models_callback(ctx, param, value):
    return load_packages(value, Model)


def dataset_callback(ctx, param, value):
    instances = load_packages(value, Dataset)
    default_name = list(instances.keys())[0]
    if len(instances) > 1:
        dataset_name = click.prompt('Multiple datasets detected, please specify',
                                    type=click.Choice(list(instances.keys())), default=default_name)
        return instances.get(dataset_name)
    return instances.get(default_name)


def dataset_pkg_option(func):
    @click.option("--dataset-pkg", "dataset", callback=dataset_callback, default="dataset", show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def models_pkg_option(func):
    @click.option("--models-pkg", "models", callback=models_callback, default="models", show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def split_path_option(func):
    @click.option("--split-path", callback=split_input_callback, type=click.Path(exists=True))
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


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
@click.option("--latest-model-eval-path", callback=latest_model_eval_input_callback, type=click.Path(exists=True))
@click.option("--compare-model-eval-path", callback=compare_model_eval_input_callback, type=click.Path(exists=True))
def server_start_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str, dataset_eval_path: str,
                     latest_model_eval_path: str, compare_model_eval_path: str):
    """Start server."""
    start_server(models, dataset, split_path, dataset_eval_path, latest_model_eval_path, compare_model_eval_path)


@click.group(name='model')
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
@models_group.command(name='eval')
@models_pkg_option
@dataset_pkg_option
@split_path_option
@click.option("-s", "--split-type", default="validation", type=click.Choice(["test", "validation"]), show_default=True)
@click.option("-o", "--output-path", callback=models_eval_output_callback)
@click.option("-m", "--model")
def models_eval_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str, split_type: str, output_path: str,
                    model: str):
    """Evaluate models using test/validation split."""
    evaluate_models(models, dataset, split_path, split_type, output_path, model)


@models_group.command(name='train')
@dataset_pkg_option
@models_pkg_option
@split_path_option
@click.option("-m", "--model")
def models_train_cmd(models: Dict[str, Model], dataset: Dataset, split_path: str, model: str):
    """Train models using train split."""
    train_models(models, dataset, split_path, model)


# DATASET GROUP
@dataset_group.command(name='split')
@dataset_pkg_option
@click.option("-o", "--output-path", callback=split_output_callback)
def dataset_split_cmd(dataset: Dataset, output_path: str):
    """Create train/validation/test split."""
    split_dataset(dataset, output_path)


@dataset_group.command(name='eval')
@dataset_pkg_option
@split_path_option
@click.option("-o", "--output-path", callback=dataset_eval_output_callback)
@click.option("--method", default="pymde", type=click.Choice(["pymde", "tsne"]), show_default=True)
def dataset_eval_cmd(dataset: Dataset, split_path: str, output_path: str, method: str):
    """Create dataset embeddings using train split."""
    evaluate_dataset(dataset, split_path, output_path, method)
