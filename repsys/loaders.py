import importlib
import inspect
import pkgutil
import logging
import sys
from typing import Dict, List, Text

from repsys.model import Model
from repsys.dataset import Dataset
from repsys.utils import get_subclasses

logger = logging.getLogger(__name__)


class ClassLoader:
    def __init__(self, cls) -> None:
        self.cls = cls
        self.instances = {}

    def _create_instance(self, x) -> None:
        if inspect.isclass(x):
            instance = x()

        if isinstance(instance, self.cls):
            name = instance.name()
            self.instances[name] = instance
        else:
            raise Exception("Invalid class instance.")

    def _import_submodules(self, package_path) -> None:
        package = importlib.import_module(package_path)

        if not getattr(package, "__path__", None):
            return

        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + "." + name
            importlib.import_module(full_name)

            if is_pkg:
                self._import_submodules(full_name)

    def register_package(self, package_path) -> None:
        try:
            self._import_submodules(package_path)
        except ImportError:
            logger.exception(f"Failed to register package '{package_path}'.")
            sys.exit(1)

        for x in get_subclasses(self.cls):
            if not inspect.isabstract(x):
                self._create_instance(x)


def load_models_pkg(models_pkg) -> List[Model]:
    logger.debug("Loading models package ...")
    model_loader = ClassLoader(Model)
    model_loader.register_package(models_pkg)

    if len(model_loader.instances) == 0:
        raise Exception("At least one model must be defined.")

    models = model_loader.instances.values()

    return models


def load_dataset_pkg(dataset_pkg) -> Dataset:
    logger.debug("Loading dataset package ...")
    dataset_loader = ClassLoader(Dataset)
    dataset_loader.register_package(dataset_pkg)

    if len(dataset_loader.instances) != 1:
        raise Exception("There must be exactly one dataset defined.")

    dataset = list(dataset_loader.instances.values())[0]

    return dataset
