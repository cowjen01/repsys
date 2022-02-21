import importlib
import inspect
import logging
import pkgutil
import re
import sys
from typing import Dict

from repsys.dataset import Dataset
from repsys.errors import PackageLoaderError
from repsys.helpers import get_subclasses
from repsys.model import Model

logger = logging.getLogger(__name__)

instance_name_regex = r"^[a-z0-9]*$"


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
                raise PackageLoaderError("Invalid class instance.")

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


def validate_model_instances(instances: Dict[str, Model]):
    if len(instances) == 0:
        raise PackageLoaderError("At least one model must be defined.")

    for model in instances.values():
        if not re.search(instance_name_regex, model.name()):
            raise PackageLoaderError("Model's name must contain only alphanumeric characters.")


def validate_dataset_instances(instances: Dict[str, Dataset]):
    if len(instances) != 1:
        raise PackageLoaderError("There must be exactly one dataset defined.")

    if not re.search(instance_name_regex, list(instances.values())[0].name()):
        raise PackageLoaderError("Dataset's name must contain only alphanumeric characters.")


def load_models_pkg(models_pkg) -> Dict[str, Model]:
    logger.debug("Loading models package ...")
    model_loader = ClassLoader(Model)
    model_loader.register_package(models_pkg)
    instances = model_loader.instances

    validate_model_instances(instances)

    return instances


def load_dataset_pkg(dataset_pkg) -> Dataset:
    logger.debug("Loading dataset package ...")
    dataset_loader = ClassLoader(Dataset)
    dataset_loader.register_package(dataset_pkg)
    instances = dataset_loader.instances

    validate_dataset_instances(instances)

    return list(instances.values())[0]
