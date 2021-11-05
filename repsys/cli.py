import click

from repsys.models import Model

from .server import run_server
from .loader import ClassLoader
from .constants import DEFAULT_SERVER_PORT


@click.group()
def repsys():
    """Repsys client for managing of recommender models."""
    pass


@repsys.command()
@click.option(
    "-m", "--models", "package", default="models.models", show_default=True
)
def train(package):
    loader = ClassLoader(Model)
    loader.register_package(package)
    loader.instances.get("KNN10").name()


@repsys.command()
@click.option(
    "-p", "--port", "port", default=DEFAULT_SERVER_PORT, show_default=True
)
@click.option(
    "-m", "--models", "package", default="models.models", show_default=True
)
def server(port, package):
    """Start Repsys server."""
    loader = ClassLoader(Model)
    loader.register_package(package)

    run_server(port=port, model_loader=loader)
