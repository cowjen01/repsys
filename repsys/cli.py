import click

from .models import Model
from .server import run_server
from .loader import ClassLoader
from .dataset import Dataset
from .constants import DEFAULT_SERVER_PORT


@click.group()
def repsys():
    """Repsys client for managing of recommender models."""
    pass


@repsys.command()
@click.option(
    "-p", "--port", "port", default=DEFAULT_SERVER_PORT, show_default=True
)
@click.option("-m", "--models", "models_package", default="models", show_default=True)
@click.option(
    "-d", "--dataset", "dataset_package", default="dataset", show_default=True
)
def server(port, models_package, dataset_package):
    """Start Repsys server."""

    dataset_loader = ClassLoader(Dataset)
    dataset_loader.register_package(dataset_package)

    dataset = list(dataset_loader.instances.values())[0]
    dataset.load_dataset()

    model_loader = ClassLoader(Model)
    model_loader.register_package(models_package)

    for model in model_loader.instances.values():
        model.dataset = dataset
        model.fit()

    run_server(port=port, models=model_loader.instances, dataset=dataset)
