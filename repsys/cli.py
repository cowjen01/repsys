import logging
from typing import Dict

import click

from repsys.core import train_models, evaluate_dataset, start_server, split_dataset, evaluate_models
from repsys.dataset import Dataset
from repsys.helpers import *
from repsys.loaders import load_packages
from repsys.model import Model

logger = logging.getLogger(__name__)


def checkpoints_dir_callback(ctx, param, value):
    if not value:
        return checkpoints_dir_path()
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


def checkpoints_dir_option(func):
    @click.option("--checkpoints-dir", callback=checkpoints_dir_callback, type=click.Path(exists=True))
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def repsys_group(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    create_checkpoints_dir()


@repsys_group.command(name='server')
@models_pkg_option
@dataset_pkg_option
@checkpoints_dir_option
def server_start_cmd(models: Dict[str, Model], dataset: Dataset, checkpoints_dir: str):
    """Start server."""
    start_server(models, dataset, checkpoints_dir)


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
@checkpoints_dir_option
@click.option("-s", "--split-type", default="validation", type=click.Choice(["test", "validation"]), show_default=True)
@click.option("-m", "--model-name")
def models_eval_cmd(models: Dict[str, Model], dataset: Dataset, split_type: str, checkpoints_dir: str, model_name: str):
    evaluate_models(models, dataset, checkpoints_dir, split_type, model_name)


@models_group.command(name='train')
@dataset_pkg_option
@models_pkg_option
@checkpoints_dir_option
@click.option("-m", "--model-name")
def models_train_cmd(models: Dict[str, Model], dataset: Dataset, checkpoints_dir: str, model_name: str):
    train_models(models, dataset, checkpoints_dir, model_name)


# DATASET GROUP
@dataset_group.command(name='split')
@dataset_pkg_option
@checkpoints_dir_option
def dataset_split_cmd(dataset: Dataset, checkpoints_dir: str):
    split_dataset(dataset, checkpoints_dir)


@dataset_group.command(name='eval')
@dataset_pkg_option
@checkpoints_dir_option
@click.option("--method", default="pymde", type=click.Choice(["pymde", "tsne", "custom"]), show_default=True)
def dataset_eval_cmd(dataset: Dataset, checkpoints_dir: str, method: str):
    evaluate_dataset(dataset, checkpoints_dir, method)
