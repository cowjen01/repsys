import click

from .server import app


@click.group()
def repsys():
    """Repsys client for managing of recommender models."""
    pass


@repsys.command()
@click.option("-p", "--port", "port", default=8080, show_default=True)
def server(port):
    """Start Repsys server."""
    app.run(host="0.0.0.0", port=port, debug=True)


def main():
    repsys(prog_name="repsys")
