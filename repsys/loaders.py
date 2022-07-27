import importlib
import inspect
import logging
import pkgutil
import re
import sys
from typing import Dict, Any

from repsys.errors import PackageLoaderError
from repsys.helpers import get_subclasses

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


def validate_instances(instances: Dict[str, Any]):
    if len(instances) == 0:
        raise PackageLoaderError("At least one class must be defined.")

    for inst in instances.values():
        if not re.search(r"^[a-z0-9]*$", inst.name()):
            raise PackageLoaderError("Name of the class must contain only alphanumeric characters.")


def load_packages(pkg, cls) -> Dict[str, Any]:
    logger.debug(f"Loading '{cls.__name__}' package ...")
    loader = ClassLoader(cls)
    loader.register_package(pkg)

    validate_instances(loader.instances)

    return loader.instances
